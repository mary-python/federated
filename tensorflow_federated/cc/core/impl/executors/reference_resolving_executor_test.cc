/* Copyright 2021, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

// Unit tests for the ReferenceResolvingExecutor.
//
// IMPORTANT: many of the `v0::Value` protocol buffer messages used in the unit
// tests in this file are not well-formed from the view of the entire execution
// stack.  Particularly `federated_language::Computation` message fields that
// are not used by the ReferenceResolvingExecutor are often ellided to assert
// that they are not dependend on. This generally means the test protos are only
// valid because the child executor is mocked out and returns a hardcoded
// result, and should not be used a reference for how a real
// `federated_language::Computation` protocol buffer message should look.

#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor_test_base.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {
namespace {

constexpr char kTestPlacement[] = "TEST";

using ::absl::StatusCode;
using ::testing::_;
using testing::BlockComputation;
using testing::ComputationV;
using testing::DataComputation;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using testing::EqualsProto;
using ::testing::Exactly;
using ::testing::HasSubstr;
using testing::IntrinsicComputation;
using testing::LambdaComputation;
using ::testing::Optional;
using testing::PlacementComputation;
using testing::ReferenceComputation;
using ::testing::Return;
using testing::SelectionComputation;
using ::testing::StrictMock;
using testing::StructComputation;
using testing::StructV;

MATCHER_P(HasValueId, expected_id,
          absl::StrCat("matches ValueId ", expected_id)) {
  *result_listener << "where the ValueId is " << arg.ref();
  return arg.ref() == expected_id;
}

class ReferenceResolvingExecutorTest : public ExecutorTestBase {
 public:
  ReferenceResolvingExecutorTest() {
    test_executor_ =
        tensorflow_federated::CreateReferenceResolvingExecutor(mock_executor_);
  }

 protected:
  std::shared_ptr<StrictMock<MockExecutor>> mock_executor_ =
      std::make_shared<StrictMock<MockExecutor>>();
};

TEST_F(ReferenceResolvingExecutorTest, CreateValueChildExecutorError) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(value_pb)))
      .WillOnce([]() { return absl::InternalError("test"); });
  EXPECT_THAT(test_executor_->CreateValue(value_pb),
              StatusIs(StatusCode::kInternal, "test"));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueTensor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  constexpr int kNumValues = 3;
  mock_executor_->ExpectCreateValue(value_pb, Exactly(kNumValues));
  for (int i = 0; i < kNumValues; ++i) {
    ASSERT_THAT(test_executor_->CreateValue(value_pb),
                IsOkAndHolds(HasValueId(i)));
  }
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueSequence) {
  v0::Value sequence_val_pb;
  *sequence_val_pb.mutable_sequence() = v0::Value::Sequence();
  constexpr int kNumValues = 3;
  mock_executor_->ExpectCreateValue(sequence_val_pb, Exactly(kNumValues));
  for (int i = 0; i < kNumValues; ++i) {
    ASSERT_THAT(test_executor_->CreateValue(sequence_val_pb),
                IsOkAndHolds(HasValueId(i)));
  }
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueFederatedTensor) {
  v0::Value federated_value_pb;
  v0::Value::Federated* federated_pb = federated_value_pb.mutable_federated();
  federated_language::FederatedType* type_pb = federated_pb->mutable_type();
  type_pb->set_all_equal(false);
  type_pb->mutable_placement()->mutable_value()->set_uri(kTestPlacement);
  federated_language::TensorType* tensor_type =
      type_pb->mutable_member()->mutable_tensor();
  tensor_type->set_dtype(federated_language::DataType::DT_FLOAT);
  constexpr int kNumClients = 3;
  for (int i = 0; i < kNumClients; ++i) {
    federated_language::Array array_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    v0::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    *federated_pb->add_value() = value_pb;
  }
  mock_executor_->ExpectCreateValue(federated_value_pb);
  EXPECT_THAT(test_executor_->CreateValue(federated_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueStructOfTensor) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3.0}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  v0::Value struct_value_pb = StructV({value1_pb, value2_pb, value3_pb});
  for (const v0::Value::Struct::Element& element_pb :
       struct_value_pb.struct_().element()) {
    const v0::Value& tensor_value_pb = element_pb.value();
    mock_executor_->ExpectCreateValue(tensor_value_pb);
  }
  // Expect ID 0, the first for ReferenceResolvingExecutor (ignoring the
  // IDs of the inner child executor);
  EXPECT_THAT(test_executor_->CreateValue(struct_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueNestedStructOfTensor) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3.0}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  v0::Value struct_value_pb =
      StructV({StructV({value1_pb, value2_pb}), value3_pb});
  // Expect three calls to CreateValue() on the inner mock, once for each
  // element of the struct.
  mock_executor_->ExpectCreateValue(value1_pb);
  mock_executor_->ExpectCreateValue(value2_pb);
  mock_executor_->ExpectCreateValue(value3_pb);
  // Expect ID 0, the second for ReferenceResolvingExecutor (the inner struct
  // and the child tensor values do not increase the count, they are internal
  // only);
  EXPECT_THAT(test_executor_->CreateValue(struct_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueFederatedStructOfTensor) {
  v0::Value federated_value_pb;
  v0::Value::Federated* federated_pb = federated_value_pb.mutable_federated();
  federated_language::FederatedType* type_pb = federated_pb->mutable_type();
  type_pb->set_all_equal(false);
  type_pb->mutable_placement()->mutable_value()->set_uri(kTestPlacement);
  federated_language::StructType* struct_type =
      type_pb->mutable_member()->mutable_struct_();
  constexpr int kNumFields = 3;
  for (int i = 0; i < kNumFields; ++i) {
    federated_language::StructType::Element* element_pb =
        struct_type->add_element();
    element_pb->mutable_value()->mutable_tensor()->set_dtype(
        federated_language::DataType::DT_FLOAT);
  }
  constexpr int kNumClients = 3;
  for (int i = 0; i < kNumClients; ++i) {
    federated_language::Array array1_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i}));
    v0::Value value1_pb;
    value1_pb.mutable_array()->Swap(&array1_pb);
    federated_language::Array array2_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i + 1}));
    v0::Value value2_pb;
    value2_pb.mutable_array()->Swap(&array2_pb);
    federated_language::Array array3_pb = TFF_ASSERT_OK(
        testing::CreateArray(federated_language::DataType::DT_INT32,
                             testing::CreateArrayShape({}), {i + 2}));
    v0::Value value3_pb;
    value3_pb.mutable_array()->Swap(&array3_pb);

    *federated_pb->add_value() = StructV({value1_pb, value2_pb, value3_pb});
  }
  mock_executor_->ExpectCreateValue(federated_value_pb);
  EXPECT_THAT(test_executor_->CreateValue(federated_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationTensorflow) {
  v0::Value tensorflow_value_pb;
  tensorflow_value_pb.mutable_computation()->mutable_tensorflow();
  mock_executor_->ExpectCreateValue(tensorflow_value_pb);
  EXPECT_THAT(test_executor_->CreateValue(tensorflow_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationXla) {
  v0::Value xla_value_pb;
  xla_value_pb.mutable_computation()->mutable_xla();
  mock_executor_->ExpectCreateValue(xla_value_pb);
  EXPECT_THAT(test_executor_->CreateValue(xla_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationData) {
  v0::Value data_comp_pb = ComputationV(DataComputation("test_data_uri"));
  mock_executor_->ExpectCreateValue(data_comp_pb);
  EXPECT_THAT(test_executor_->CreateValue(data_comp_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationIntrinsic) {
  v0::Value intrinsic_comp_pb =
      ComputationV(IntrinsicComputation("test_intrinsic_uri"));
  mock_executor_->ExpectCreateValue(intrinsic_comp_pb);
  EXPECT_THAT(test_executor_->CreateValue(intrinsic_comp_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationPlacement) {
  v0::Value placement_comp_pb =
      ComputationV(PlacementComputation("test_placement_uri"));
  mock_executor_->ExpectCreateValue(placement_comp_pb);
  EXPECT_THAT(test_executor_->CreateValue(placement_comp_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest, NoArgLambda) {
  federated_language::Computation data_pb = DataComputation("test_data_uri");
  v0::Value lambda_pb = ComputationV(LambdaComputation(std::nullopt, data_pb));
  auto create_result = test_executor_->CreateValue(lambda_pb);
  EXPECT_THAT(create_result, IsOkAndHolds(HasValueId(0)));
  // Expect that the CreateCall causes the lambda to be evaluated and embedded
  // in the child executor.
  ValueId mock_value_id =
      mock_executor_->ExpectCreateValue(ComputationV(data_pb));
  auto call_result =
      test_executor_->CreateCall(create_result.value().ref(), std::nullopt);
  EXPECT_THAT(call_result, IsOkAndHolds(HasValueId(1)));
  // Expect the materialize fetches the computation result from the child
  // executor.
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  mock_executor_->ExpectMaterialize(mock_value_id, value_pb);
  ExpectMaterialize(call_result.value(), value_pb);
}

TEST_F(ReferenceResolvingExecutorTest, OneArgLambda) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  v0::Value lambda_pb = ComputationV(
      LambdaComputation("test_arg", ReferenceComputation("test_arg")));
  auto create_lambda_result = test_executor_->CreateValue(lambda_pb);
  EXPECT_THAT(create_lambda_result, IsOkAndHolds(HasValueId(0)));
  // Create the argument that will be passed to the lambda.
  ValueId arg_child_id = mock_executor_->ExpectCreateValue(value_pb);
  auto create_arg_result = test_executor_->CreateValue(value_pb);
  EXPECT_THAT(create_arg_result, IsOkAndHolds(HasValueId(1)));
  // Expect that the CreateCall to be ID 2, since both the Reference and the
  // Lambda have taken IDs already.
  auto call_result = test_executor_->CreateCall(
      create_lambda_result.value().ref(), create_arg_result.value().ref());
  EXPECT_THAT(call_result, IsOkAndHolds(HasValueId(2)));
  // Expect the materialize fetches the computation result from the child
  // executor.
  federated_language::Array array_result_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_result_pb;
  value_result_pb.mutable_array()->Swap(&array_result_pb);
  mock_executor_->ExpectMaterialize(arg_child_id, value_result_pb);
  ExpectMaterialize(call_result.value(), value_result_pb);
}

TEST_F(ReferenceResolvingExecutorTest, LambdaStructArgumentLazilyEmbedded) {
  v0::Value lambda_pb = ComputationV(
      LambdaComputation("test_arg", ReferenceComputation("test_arg")));
  auto create_lambda_result = test_executor_->CreateValue(lambda_pb);
  EXPECT_THAT(create_lambda_result, IsOkAndHolds(HasValueId(0)));
  // Create the argument struct that will be passed to the lambda. It will
  // be lazily embedded in the child executor.
  std::vector<OwnedValueId> arg_slots;
  std::vector<ValueId> element_ids;
  std::vector<ValueId> element_child_ids;
  for (int i = 0; i < 3; ++i) {
    federated_language::Array array_pb = TFF_ASSERT_OK(testing::CreateArray(
        federated_language::DataType::DT_FLOAT, testing::CreateArrayShape({}),
        {static_cast<float>(i)}));
    v0::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    ValueId child_id = mock_executor_->ExpectCreateValue(value_pb);
    auto create_arg_result = test_executor_->CreateValue(value_pb);
    EXPECT_THAT(create_arg_result, IsOkAndHolds(HasValueId(i + 1)));
    element_ids.push_back(create_arg_result.value().ref());
    element_child_ids.push_back(child_id);
    arg_slots.emplace_back(std::move(create_arg_result).value());
  }
  auto create_arg_result = test_executor_->CreateStruct(element_ids);
  EXPECT_THAT(create_arg_result, IsOkAndHolds(HasValueId(4)));
  // Expect the CreateCall to cause embedding the struct in the child executor.
  ValueId struct_child_id =
      mock_executor_->ExpectCreateStruct(element_child_ids);
  // Expect that the CreateCall to be ID 5. 0 for the lambda, 1-3 the argument
  // elements, 4 for the argument struct.
  auto call_result = test_executor_->CreateCall(
      create_lambda_result.value().ref(), create_arg_result.value().ref());
  EXPECT_THAT(call_result, IsOkAndHolds(HasValueId(5)));
  // Expect the materialize fetches the computation result from the child
  // executor.
  federated_language::Array array_result_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_result_pb;
  value_result_pb.mutable_array()->Swap(&array_result_pb);
  mock_executor_->ExpectMaterialize(struct_child_id, value_result_pb);
  ExpectMaterialize(call_result.value(), value_result_pb);
}

TEST_F(ReferenceResolvingExecutorTest,
       LambdaArgumentScopeHidesBlockNamedValue) {
  federated_language::Computation data_pb = DataComputation("test_data_uri");
  v0::Value lambda_pb = ComputationV(BlockComputation(
      {{"test_arg", data_pb},
       {"test_lambda",
        LambdaComputation("test_arg", ReferenceComputation("test_arg"))}},
      ReferenceComputation("test_lambda")));
  LOG(INFO) << lambda_pb.ShortDebugString();
  // Expect the data local to be created on the child executor.
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(ComputationV(data_pb))))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 100); });
  EXPECT_CALL(*mock_executor_, Dispose(100));
  auto create_lambda_result = test_executor_->CreateValue(lambda_pb);
  EXPECT_THAT(create_lambda_result, IsOkAndHolds(HasValueId(0)));
  // Create the argument that will be passed to the lambda.
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(value_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 200); });
  EXPECT_CALL(*mock_executor_, Dispose(200));
  auto create_arg_result = test_executor_->CreateValue(value_pb);
  EXPECT_THAT(create_arg_result, IsOkAndHolds(HasValueId(1)));
  // Expect that the CreateCall to be ID 2, since both the Reference and the
  // Lambda have taken IDs already.
  auto call_result = test_executor_->CreateCall(
      create_lambda_result.value().ref(), create_arg_result.value().ref());
  EXPECT_THAT(call_result, IsOkAndHolds(HasValueId(2)));
  // Expect the materialize fetches the computation result from the child
  // executor for the argument, not the data local with the same name.
  EXPECT_CALL(*mock_executor_, Materialize(200, _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(test_executor_->Materialize(call_result.value()), IsOk());
}

TEST_F(ReferenceResolvingExecutorTest, LambdaArgumentToInstrinsicIsEmbedded) {
  v0::Value intrinsic_pb = ComputationV(IntrinsicComputation("test_intrinsic"));
  v0::Value lambda_arg_pb = ComputationV(
      LambdaComputation("test_arg", ReferenceComputation("test_arg")));
  v0::Value lambda_pb;
  *lambda_pb.mutable_computation()->mutable_call()->mutable_function() =
      intrinsic_pb.computation();
  *lambda_pb.mutable_computation()->mutable_call()->mutable_argument() =
      lambda_arg_pb.computation();
  // Expect create value on a Call to evaluate the function and argument, then
  // create a call.
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(intrinsic_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 100); });
  EXPECT_CALL(*mock_executor_, Dispose(100));
  // Expect the lambda argument to be embedded in the child executor.
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(lambda_arg_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 200); });
  EXPECT_CALL(*mock_executor_, Dispose(200));
  EXPECT_CALL(*mock_executor_, CreateCall(100, Optional(200)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 300); });
  EXPECT_CALL(*mock_executor_, Dispose(300));
  auto create_lambda_result = test_executor_->CreateValue(lambda_pb);
  EXPECT_THAT(create_lambda_result, IsOkAndHolds(HasValueId(0)));
  // Expect the materialize on the call to be pushed down.
  EXPECT_CALL(*mock_executor_, Materialize(300, _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(test_executor_->Materialize(create_lambda_result.value()),
              IsOk());
}

TEST_F(ReferenceResolvingExecutorTest,
       CreateValueComputationBlockSingleLocalUsingReference) {
  v0::Value block_value_pb = ComputationV(
      BlockComputation({{"test_ref", DataComputation("test_data_uri")}},
                       ReferenceComputation("test_ref")));
  // Expect CreateValue to be called on each local in the block, and delegate
  // the locals to the child executor if necessary.
  constexpr ValueId child_id = 3;
  EXPECT_CALL(
      *mock_executor_,
      CreateValue(::testing::Property(
          &v0::Value::computation,
          EqualsProto(block_value_pb.computation().block().local(0).value()))))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, child_id); });
  EXPECT_CALL(*mock_executor_, Dispose(child_id));
  EXPECT_THAT(test_executor_->CreateValue(block_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest,
       CreateValueComputationBlockUniqueLocals) {
  v0::Value block_value_pb = ComputationV(
      BlockComputation({{"test_ref", DataComputation("test_data_uri")},
                        {"test_ref2", DataComputation("test_data_uri2")}},
                       ReferenceComputation("test_ref")));
  ValueId child_id = 3;
  for (const federated_language::Block::Local& local_pb :
       block_value_pb.computation().block().local()) {
    EXPECT_CALL(*mock_executor_, Dispose(child_id));
    EXPECT_CALL(*mock_executor_,
                CreateValue(::testing::Property(&v0::Value::computation,
                                                EqualsProto(local_pb.value()))))
        .WillOnce([this, child_id]() {
          return OwnedValueId(mock_executor_, child_id);
        });
    ++child_id;
  }
  EXPECT_THAT(test_executor_->CreateValue(block_value_pb),
              IsOkAndHolds(HasValueId(0)));
}

TEST_F(ReferenceResolvingExecutorTest,
       CreateValueComputationBlockDulicatedLocals) {
  // Expect that the nested/later scope in the second local is used for the
  // reference.
  v0::Value block_value_pb = ComputationV(
      BlockComputation({{"test_ref", DataComputation("test_data_uri")},
                        {"test_ref", DataComputation("test_data_uri2")}},
                       ReferenceComputation("test_ref")));
  ValueId child_id = 3;
  for (const federated_language::Block::Local& local_pb :
       block_value_pb.computation().block().local()) {
    EXPECT_CALL(*mock_executor_, Dispose(child_id));
    EXPECT_CALL(*mock_executor_,
                CreateValue(::testing::Property(&v0::Value::computation,
                                                EqualsProto(local_pb.value()))))
        .WillOnce([this, child_id]() {
          return OwnedValueId(mock_executor_, child_id);
        });
    ++child_id;
  }
  auto create_result = test_executor_->CreateValue(block_value_pb);
  EXPECT_THAT(create_result, IsOkAndHolds(HasValueId(0)));
  // Expect that the mock executor materializes the value with ID 4 (the second
  // local, the first is ID 3).
  EXPECT_CALL(*mock_executor_, Materialize(4, _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(test_executor_->Materialize(create_result.value()), IsOk());
}

TEST_F(ReferenceResolvingExecutorTest,
       EvaluateBlockLocalReferencesPreviousLocal) {
  // Expect that the nested/later scope in the second local is used for the
  // reference.
  v0::Value block_value_pb = ComputationV(
      BlockComputation({{"test_ref1", DataComputation("test_data_uri")},
                        {"test_ref2", ReferenceComputation("test_ref1")}},
                       ReferenceComputation("test_ref2")));
  // We only create expectations for the first local, because the second
  // local simply reference the first.
  constexpr ValueId local_id = 3;
  EXPECT_CALL(*mock_executor_, Dispose(local_id));
  EXPECT_CALL(
      *mock_executor_,
      CreateValue(::testing::Property(
          &v0::Value::computation,
          EqualsProto(block_value_pb.computation().block().local(0).value()))))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, local_id); });
  auto create_result = test_executor_->CreateValue(block_value_pb);
  EXPECT_THAT(create_result, IsOkAndHolds(HasValueId(0)));
  // Expect that the mock executor materializes the value with ID 3 (the
  // second local, which is just a reference to the first which is ID 3).
  EXPECT_CALL(*mock_executor_, Materialize(local_id, _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(test_executor_->Materialize(create_result.value()), IsOk());
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationReferenceMissing) {
  v0::Value block_value_pb = ComputationV(BlockComputation(
      {{"test_ref", DataComputation("test_data_uri")},
       {"test_ref2", StructComputation({DataComputation("test_data_uri2"),
                                        DataComputation("test_data_uri3")})}},
      ReferenceComputation("test_ref3")));
  ValueId child_id = 3;
  EXPECT_CALL(*mock_executor_, CreateValue(_))
      .Times(3)
      .WillRepeatedly([this, &child_id]() {
        return OwnedValueId(mock_executor_, ++child_id);
      });
  EXPECT_CALL(*mock_executor_, Dispose(_)).Times(3);
  EXPECT_THAT(
      test_executor_->CreateValue(block_value_pb),
      StatusIs(StatusCode::kNotFound,
               "Could not find reference [test_ref3] while searching scope: "
               "[]->[test_ref=V]->[test_ref2=<V>]"));
}

TEST_F(ReferenceResolvingExecutorTest, CreateCallFailsNonFunction) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value struct_value_pb = StructV({value1_pb, value2_pb});
  EXPECT_CALL(*mock_executor_, CreateValue(_))
      .Times(2)
      .WillRepeatedly([this]() { return OwnedValueId(mock_executor_, 0); });
  EXPECT_CALL(*mock_executor_, Dispose(_)).Times(2);
  auto result = test_executor_->CreateValue(struct_value_pb);
  ASSERT_THAT(result, IsOkAndHolds(HasValueId(0)));
  EXPECT_THAT(
      test_executor_->CreateCall(result.value(), std::nullopt),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr(
                   "Received value type [STRUCTURE] which is not a function")));
}

TEST_F(ReferenceResolvingExecutorTest, CreateCallNoArgComp) {
  v0::Value no_arg_computation_pb;
  no_arg_computation_pb.mutable_computation()->mutable_tensorflow();
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(no_arg_computation_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 0); });
  EXPECT_CALL(*mock_executor_, Dispose(0));
  auto result = test_executor_->CreateValue(no_arg_computation_pb);
  ASSERT_THAT(result, IsOkAndHolds(HasValueId(0)));
  EXPECT_CALL(*mock_executor_, CreateCall(0, Eq(std::nullopt)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 1); });
  EXPECT_CALL(*mock_executor_, Dispose(1));
  EXPECT_THAT(test_executor_->CreateCall(result.value().ref(), std::nullopt),
              IsOkAndHolds(HasValueId(1)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateCallNoArgCompWithArg) {
  v0::Value no_arg_computation_pb;
  no_arg_computation_pb.mutable_computation()->mutable_tensorflow();
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(no_arg_computation_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 0); });
  EXPECT_CALL(*mock_executor_, Dispose(0));
  auto result = test_executor_->CreateValue(no_arg_computation_pb);
  ASSERT_THAT(result, IsOkAndHolds(HasValueId(0)));
  // Create an argument.
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(value_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 1); });
  EXPECT_CALL(*mock_executor_, Dispose(1));
  auto arg_result = test_executor_->CreateValue(value_pb);
  ASSERT_THAT(arg_result, IsOkAndHolds(HasValueId(1)));
  // Create the call.
  // Note: we don't error at this level because there is no type checking of the
  // function signature to see if bindings are valid.
  EXPECT_CALL(*mock_executor_, CreateCall(0, Optional(1))).WillOnce([this]() {
    return OwnedValueId(mock_executor_, 2);
  });
  EXPECT_CALL(*mock_executor_, Dispose(2));
  EXPECT_THAT(test_executor_->CreateCall(result.value().ref(),
                                         arg_result.value().ref()),
              IsOkAndHolds(HasValueId(2)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateCallSingleArg) {
  v0::Value no_arg_computation_pb;
  no_arg_computation_pb.mutable_computation()->mutable_tensorflow();
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(no_arg_computation_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 0); });
  EXPECT_CALL(*mock_executor_, Dispose(0));
  auto result = test_executor_->CreateValue(no_arg_computation_pb);
  ASSERT_THAT(result, IsOkAndHolds(HasValueId(0)));
  // Create an argument.
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(value_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 1); });
  EXPECT_CALL(*mock_executor_, Dispose(1));
  auto arg_result = test_executor_->CreateValue(value_pb);
  ASSERT_THAT(arg_result, IsOkAndHolds(HasValueId(1)));
  // Create the call.
  EXPECT_CALL(*mock_executor_, CreateCall(0, Optional(1))).WillOnce([this]() {
    return OwnedValueId(mock_executor_, 2);
  });
  EXPECT_CALL(*mock_executor_, Dispose(2));
  EXPECT_THAT(test_executor_->CreateCall(result.value().ref(),
                                         arg_result.value().ref()),
              IsOkAndHolds(HasValueId(2)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateCallLazyStructMultiArg) {
  v0::Value no_arg_computation_pb;
  no_arg_computation_pb.mutable_computation()->mutable_tensorflow();
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(no_arg_computation_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 0); });
  EXPECT_CALL(*mock_executor_, Dispose(0));
  auto result = test_executor_->CreateValue(no_arg_computation_pb);
  ASSERT_THAT(result, IsOkAndHolds(HasValueId(0)));
  // Create arguments.
  std::vector<OwnedValueId> args;
  for (int i = 0; i < 2; ++i) {
    federated_language::Array array_pb = TFF_ASSERT_OK(testing::CreateArray(
        federated_language::DataType::DT_FLOAT, testing::CreateArrayShape({}),
        {static_cast<float>(i)}));
    v0::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(value_pb)))
        .WillOnce([this, i]() { return OwnedValueId(mock_executor_, i + 1); });
    EXPECT_CALL(*mock_executor_, Dispose(i + 1));
    auto arg_result = test_executor_->CreateValue(value_pb);
    ASSERT_THAT(arg_result, IsOkAndHolds(HasValueId(i + 1)));
    args.emplace_back(std::move(arg_result).value());
  }
  std::vector<ValueId> struct_elements = {args[0].ref(), args[1].ref()};
  auto arg_struct_result = test_executor_->CreateStruct(struct_elements);
  // Create the call. Expect that the arg is now created as a struct in the
  // child executor before the call.
  EXPECT_CALL(*mock_executor_, CreateStruct(ElementsAreArray(struct_elements)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 3); });
  EXPECT_CALL(*mock_executor_, Dispose(3));
  EXPECT_CALL(*mock_executor_, CreateCall(0, Optional(3))).WillOnce([this]() {
    return OwnedValueId(mock_executor_, 4);
  });
  EXPECT_CALL(*mock_executor_, Dispose(4));
  EXPECT_THAT(test_executor_->CreateCall(result.value().ref(),
                                         arg_struct_result.value().ref()),
              IsOkAndHolds(HasValueId(4)));
}

TEST_F(ReferenceResolvingExecutorTest, CreateStruct) {
  v0::Value struct_value_pb;
  std::vector<OwnedValueId> elements;
  std::vector<ValueId> element_ids;
  std::vector<ValueId> element_child_ids;
  for (int i = 0; i < 3; ++i) {
    federated_language::Array array_pb = TFF_ASSERT_OK(testing::CreateArray(
        federated_language::DataType::DT_FLOAT, testing::CreateArrayShape({}),
        {static_cast<float>(i)}));
    v0::Value value_pb;
    value_pb.mutable_array()->Swap(&array_pb);
    ValueId child_id = mock_executor_->ExpectCreateValue(value_pb);
    absl::StatusOr<OwnedValueId> element =
        TFF_ASSERT_OK(test_executor_->CreateValue(value_pb));
    element_ids.push_back(element.value().ref());
    element_child_ids.push_back(child_id);
    elements.emplace_back(std::move(element).value());
  }
  // We expect the fourth value (id 3) for the struct, and the struct is lazy
  // (does not immediate forward to the child executor).
  auto create_struct_result = test_executor_->CreateStruct(element_ids);
  EXPECT_THAT(create_struct_result, IsOkAndHolds(HasValueId(3)));
  // Expect that the executor now creates the struct in the child and
  // materializes it from the child.
  ValueId struct_child_id =
      mock_executor_->ExpectCreateStruct(element_child_ids);
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  mock_executor_->ExpectMaterialize(struct_child_id, value_pb);
  ExpectMaterialize(create_struct_result.value(), value_pb);
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationStruct) {
  v0::Value struct_value_pb = ComputationV(StructComputation({
      DataComputation("test_data1"),
      DataComputation("test_data2"),
      DataComputation("test_data3"),
  }));
  // Expect CreateValue calls for each element as it is embedded.
  ValueId test_id = 0;
  const uint32_t num_elements =
      struct_value_pb.computation().struct_().element_size();
  EXPECT_CALL(*mock_executor_, CreateValue(_))
      .Times(num_elements)
      .WillRepeatedly([this, &test_id]() {
        return OwnedValueId(mock_executor_, ++test_id);
      });
  EXPECT_CALL(*mock_executor_, Dispose(_)).Times(num_elements);
  // We expect a fourth value (id 3 in the child executor ) for the struct, but
  // the struct is lazily constructed (does not immediate forward to the child
  // executor).
  auto create_struct_result = test_executor_->CreateValue(struct_value_pb);
  EXPECT_THAT(create_struct_result, IsOkAndHolds(HasValueId(0)));
  // Expect that the executor create the struct in the child and materializes it
  // from the child when we call Materialize from the RRE.
  EXPECT_CALL(*mock_executor_, CreateStruct(ElementsAreArray({1, 2, 3})))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 4); });
  EXPECT_CALL(*mock_executor_, Materialize(4, _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_executor_, Dispose(4));
  EXPECT_THAT(test_executor_->Materialize(create_struct_result.value()),
              IsOk());
}

TEST_F(ReferenceResolvingExecutorTest, CreateSelection) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value struct_value_pb = StructV({value1_pb, value2_pb});
  // Expect CreateValue calls for each element as it is embedded.
  [[maybe_unused]] ValueId first_child_id =
      mock_executor_->ExpectCreateValue(value1_pb);
  ValueId second_child_id = mock_executor_->ExpectCreateValue(value2_pb);
  auto create_struct_result = test_executor_->CreateValue(struct_value_pb);
  EXPECT_THAT(create_struct_result, IsOkAndHolds(HasValueId(0)));
  auto create_selection_result =
      test_executor_->CreateSelection(create_struct_result.value(), 1);
  EXPECT_THAT(create_selection_result, IsOkAndHolds(HasValueId(1)));
  // Expect the child executor to materialize the second tensor.
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  mock_executor_->ExpectMaterialize(second_child_id, value3_pb);
  ExpectMaterialize(create_selection_result.value(), value3_pb);
}

TEST_F(ReferenceResolvingExecutorTest, CreateValueComputationSelection) {
  v0::Value selection_value_pb = ComputationV(
      SelectionComputation(StructComputation({DataComputation("test_data1"),
                                              DataComputation("test_data2")}),
                           /*index=*/1));
  // Expect CreateValue calls for each element in the struct as it is embedded.
  ValueId test_id = 0;
  const uint32_t num_elements = selection_value_pb.computation()
                                    .selection()
                                    .source()
                                    .struct_()
                                    .element_size();
  EXPECT_CALL(*mock_executor_, CreateValue(_))
      .Times(num_elements)
      .WillRepeatedly([this, &test_id]() {
        return OwnedValueId(mock_executor_, ++test_id);
      });
  EXPECT_CALL(*mock_executor_, Dispose(_)).Times(num_elements);
  // Expect no calls to the underlying executor as the struct only exists lazily
  // in the RRE and we don't need to traverse done to perform the selection.
  auto create_selection_result =
      test_executor_->CreateValue(selection_value_pb);
  EXPECT_THAT(create_selection_result, IsOkAndHolds(HasValueId(0)));
  // Expect a materialize on the second embedded value, but no CreateStruct or
  // CreateSelection as a result of materializing.
  EXPECT_CALL(*mock_executor_, Materialize(2, _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_THAT(test_executor_->Materialize(create_selection_result.value()),
              IsOk());
}

TEST_F(ReferenceResolvingExecutorTest, EvaluateSelectionOfEmbeddStruct) {
  // Calling an intrinsic will result in an embedded value. Using this to
  // create selection on an embedded value.
  v0::Value intrinsic_pb = ComputationV(IntrinsicComputation("test_intrinsic"));
  ValueId comp_child_id = mock_executor_->ExpectCreateValue(intrinsic_pb);
  auto intrinsic_result = test_executor_->CreateValue(intrinsic_pb);
  EXPECT_THAT(intrinsic_result, IsOkAndHolds(HasValueId(0)));
  // Now setup the call on the intrinsic.
  ValueId result_child_id =
      mock_executor_->ExpectCreateCall(comp_child_id, std::nullopt);
  auto call_result =
      test_executor_->CreateCall(intrinsic_result.value(), std::nullopt);
  EXPECT_THAT(call_result, IsOkAndHolds(HasValueId(1)));
  // Create a selection on the call result.
  mock_executor_->ExpectCreateSelection(result_child_id, 2);
  auto select_result = test_executor_->CreateSelection(call_result.value(), 2);
  EXPECT_THAT(select_result, IsOkAndHolds(HasValueId(2)));
}

TEST_F(ReferenceResolvingExecutorTest,
       EvaluateSelectionOfEmbeddStructChildExecutorFails) {
  // Calling an intrinsic will result in an embedded value. Using this to
  // create selection on an embedded value. In this case the child executor will
  // return an error rather.
  v0::Value intrinsic_pb = ComputationV(IntrinsicComputation("test_intrinsic"));
  ValueId intrinsic_child_id = mock_executor_->ExpectCreateValue(intrinsic_pb);
  auto intrinsic_result = test_executor_->CreateValue(intrinsic_pb);
  EXPECT_THAT(intrinsic_result, IsOkAndHolds(HasValueId(0)));
  // Now setup the call on the intrinsic.
  ValueId call_result_child_id =
      mock_executor_->ExpectCreateCall(intrinsic_child_id, std::nullopt);
  auto call_result =
      test_executor_->CreateCall(intrinsic_result.value(), std::nullopt);
  EXPECT_THAT(call_result, IsOkAndHolds(HasValueId(1)));
  // Create a selection on the call result.
  EXPECT_CALL(*mock_executor_, CreateSelection(call_result_child_id, 2))
      .WillOnce([]() { return absl::InternalError("expected test failure"); });
  EXPECT_THAT(
      test_executor_->CreateSelection(call_result.value(), 2),
      StatusIs(StatusCode::kInternal, HasSubstr("expected test failure")));
}

TEST_F(ReferenceResolvingExecutorTest,
       EvaluateSelectionFromUncalledLambdaFails) {
  v0::Value selection_value_pb = ComputationV(SelectionComputation(
      LambdaComputation("test_arg", ReferenceComputation("test_arg")),
      /*index=*/1));
  // Expect the executor returns an error because it cannot evaluate a
  // selection on a uncalled lambda.
  EXPECT_THAT(test_executor_->CreateValue(selection_value_pb),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Cannot perform selection on Lambda")));
}

TEST_F(ReferenceResolvingExecutorTest, EvaluateSelectionFailsInvalidIndex) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value struct_value_pb = StructV({value1_pb, value2_pb});
  mock_executor_->ExpectCreateValue(value1_pb);
  mock_executor_->ExpectCreateValue(value2_pb);
  auto create_struct_result = test_executor_->CreateValue(struct_value_pb);
  EXPECT_THAT(create_struct_result, IsOkAndHolds(HasValueId(0)));
  EXPECT_THAT(test_executor_->CreateSelection(create_struct_result.value(), 3),
              StatusIs(StatusCode::kNotFound,
                       HasSubstr("index [3] on structure with length [2]")));
}

TEST_F(ReferenceResolvingExecutorTest, MaterializeEmbeddedValue) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  ValueId child_id = mock_executor_->ExpectCreateValue(value_pb);
  auto result = test_executor_->CreateValue(value_pb);
  ASSERT_THAT(result, IsOk());
  mock_executor_->ExpectMaterialize(child_id, value_pb);
  ExpectMaterialize(result.value().ref(), value_pb);
}

TEST_F(ReferenceResolvingExecutorTest, MaterializeFlatStruct) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value value_pb = StructV({value1_pb, value2_pb});
  // Setup expectations for the individual tensors, ensuring that struct
  // creation is delayed until materializing.
  for (const auto& element_pb : value_pb.struct_().element()) {
    mock_executor_->ExpectCreateValue(element_pb.value());
  }
  auto result = test_executor_->CreateValue(value_pb);
  ASSERT_THAT(result, IsOk());
  // Materialize the structure will now create a struct in the child executor
  // using the two tensors it has already created.
  ValueId struct_child_id = mock_executor_->ReturnsNewValue(
      EXPECT_CALL(*mock_executor_, CreateStruct(_)));
  // Assert the child value created for the struct is cleaned up.
  mock_executor_->ExpectMaterialize(struct_child_id, value_pb);
  ExpectMaterialize(result.value(), value_pb);
}

TEST_F(ReferenceResolvingExecutorTest, MaterializeNestedStruct) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  std::vector<v0::Value> tensor_pbs = {value1_pb, value2_pb};
  v0::Value value_pb = StructV({StructV({tensor_pbs[0]}), tensor_pbs[1]});
  // Setup expectations for the individual tensors, ensuring that struct
  // creation is delayed until materializing.
  std::vector<ValueId> tensor_child_ids;
  for (const auto& tensor_pb : tensor_pbs) {
    tensor_child_ids.push_back(mock_executor_->ExpectCreateValue(tensor_pb));
  }
  auto result = test_executor_->CreateValue(value_pb);
  ASSERT_THAT(result, IsOk());
  ValueId inner_struct_child_id =
      mock_executor_->ExpectCreateStruct({tensor_child_ids[0]});
  ValueId outer_struct_child_id = mock_executor_->ExpectCreateStruct(
      {inner_struct_child_id, tensor_child_ids[1]});
  mock_executor_->ExpectMaterialize(outer_struct_child_id, value_pb);
  ExpectMaterialize(result.value(), value_pb);
}

TEST_F(ReferenceResolvingExecutorTest, MaterializeFailsOnChildFailure) {
  const v0::Value value_pb = ComputationV(
      LambdaComputation("test_arg", ReferenceComputation("test_arg")));
  auto result = test_executor_->CreateValue(value_pb);
  EXPECT_THAT(result, IsOkAndHolds(HasValueId(0)));
  ValueId child_id = mock_executor_->ExpectCreateValue(value_pb);
  EXPECT_CALL(*mock_executor_, Materialize(child_id, _))
      .WillOnce(Return(absl::InternalError("child test error")));
  v0::Value result_pb;
  EXPECT_THAT(test_executor_->Materialize(result.value().ref(), &result_pb),
              StatusIs(StatusCode::kInternal, HasSubstr("child test error")));
}

TEST_F(ReferenceResolvingExecutorTest, Dispose) {
  EXPECT_THAT(
      test_executor_->Dispose(0),
      StatusIs(StatusCode::kNotFound,
               HasSubstr("ReferenceResolvingExecutor value not found: 0")));
  const v0::Value value_pb = ComputationV(
      LambdaComputation("test_arg", ReferenceComputation("test_arg")));
  auto result = test_executor_->CreateValue(value_pb);
  EXPECT_THAT(result, IsOkAndHolds(HasValueId(0)));
  EXPECT_THAT(test_executor_->Dispose(0), IsOk());
}

TEST_F(ReferenceResolvingExecutorTest, DisposeForwardsToChildExecutor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  EXPECT_CALL(*mock_executor_, CreateValue(EqualsProto(value_pb)))
      .WillOnce([this]() { return OwnedValueId(mock_executor_, 10); });
  auto result = test_executor_->CreateValue(value_pb);
  EXPECT_THAT(result, IsOkAndHolds(HasValueId(0)));
  EXPECT_CALL(*mock_executor_, Dispose(10));
  EXPECT_THAT(test_executor_->Dispose(0), IsOk());
}

}  // namespace
}  // namespace tensorflow_federated
