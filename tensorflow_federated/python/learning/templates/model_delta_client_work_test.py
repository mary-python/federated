# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import itertools
from unittest import mock

from absl.testing import parameterized
import federated_language
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.core.backends.native import execution_contexts
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_computation
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types
from tensorflow_federated.python.core.templates import measured_process
from tensorflow_federated.python.learning import client_weight_lib
from tensorflow_federated.python.learning import loop_builder
from tensorflow_federated.python.learning.metrics import counters
from tensorflow_federated.python.learning.models import functional
from tensorflow_federated.python.learning.models import keras_utils
from tensorflow_federated.python.learning.models import model_examples
from tensorflow_federated.python.learning.models import model_weights as model_weights_lib
from tensorflow_federated.python.learning.optimizers import sgdm
from tensorflow_federated.python.learning.templates import client_works
from tensorflow_federated.python.learning.templates import model_delta_client_work


class ModelDeltaClientWorkComputationTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_initialize_has_expected_type_signature_with_tff_optimizer(
      self, weighting
  ):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    expected_state_type = federated_language.FederatedType(
        collections.OrderedDict(learning_rate=np.float32),
        federated_language.SERVER,
    )
    expected_initialize_type = federated_language.FunctionType(
        parameter=None, result=expected_state_type
    )
    self.assertTrue(
        client_work_process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type
        )
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_next_has_expected_type_signature_with_tff_optimizer(self, weighting):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    mw_type = model_weights_lib.ModelWeights(
        trainable=federated_language.to_type(
            [(np.float32, (2, 1)), np.float32]
        ),
        non_trainable=federated_language.to_type([np.float32]),
    )
    expected_param_model_weights_type = federated_language.FederatedType(
        mw_type, federated_language.CLIENTS
    )
    element_type = tensorflow_types.to_type(model_fn().input_spec)
    expected_param_data_type = federated_language.FederatedType(
        federated_language.SequenceType(element_type),
        federated_language.CLIENTS,
    )
    expected_result_type = federated_language.FederatedType(
        client_works.ClientResult(
            update=mw_type.trainable,
            update_weight=federated_language.TensorType(np.float32),
        ),
        federated_language.CLIENTS,
    )
    expected_state_type = federated_language.FederatedType(
        collections.OrderedDict(learning_rate=np.float32),
        federated_language.SERVER,
    )
    expected_measurements_type = federated_language.FederatedType(
        collections.OrderedDict(
            train=collections.OrderedDict(
                loss=np.float32, num_examples=np.int32
            )
        ),
        federated_language.SERVER,
    )
    expected_next_type = federated_language.FunctionType(
        parameter=collections.OrderedDict(
            state=expected_state_type,
            weights=expected_param_model_weights_type,
            client_data=expected_param_data_type,
        ),
        result=measured_process.MeasuredProcessOutput(
            expected_state_type,
            expected_result_type,
            expected_measurements_type,
        ),
    )
    self.assertTrue(
        client_work_process.next.type_signature.is_equivalent_to(
            expected_next_type
        )
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_get_hparams_has_expected_type_signature_with_tff_optimizer(
      self, weighting
  ):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    expected_state_type = collections.OrderedDict(learning_rate=np.float32)
    expected_hparams_type = expected_state_type
    expected_get_hparams_type = federated_language.FunctionType(
        parameter=expected_state_type, result=expected_hparams_type
    )
    self.assertTrue(
        client_work_process.get_hparams.type_signature.is_equivalent_to(
            expected_get_hparams_type
        )
    )

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_set_hparams_has_expected_type_signature_with_tff_optimizer(
      self, weighting
  ):
    optimizer = sgdm.build_sgdm(learning_rate=1.0)
    model_fn = model_examples.LinearRegression

    client_work_process = model_delta_client_work.build_model_delta_client_work(
        model_fn, optimizer, weighting
    )

    expected_state_type = collections.OrderedDict(learning_rate=np.float32)
    expected_hparams_type = expected_state_type
    expected_parameter_type = federated_language.StructType(
        [('state', expected_state_type), ('hparams', expected_hparams_type)]
    )
    expected_set_hparams_type = federated_language.FunctionType(
        parameter=expected_parameter_type, result=expected_state_type
    )
    self.assertTrue(
        client_work_process.set_hparams.type_signature.is_equivalent_to(
            expected_set_hparams_type
        )
    )

  def test_raises_with_created_model(self):
    with self.assertRaises(TypeError):
      model_delta_client_work.build_model_delta_client_work(
          model_examples.LinearRegression(),
          sgdm.build_sgdm(learning_rate=1.0),
          client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      )


def create_test_dataset() -> tf.data.Dataset:
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0.0], [0.0], [1.0], [1.0]],
      )
  )
  # Repeat the dataset 3 times with batches of 3 examples, producing 3
  # minibatches. Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(3).batch(3)


def create_test_initial_weights() -> model_weights_lib.ModelWeights:
  return model_weights_lib.ModelWeights(
      trainable=[np.zeros([2, 1], np.float32), 0.0],
      non_trainable=[0.0],
  )


def create_model():
  return model_examples.LinearRegression(feature_dim=2)


class ModelDeltaClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):
  """Tests of the client work of FedAvg using a common model and data."""

  @parameterized.named_parameters(
      (
          'dataset_reduce_noclip_uniform',
          loop_builder.LoopImplementation.DATASET_REDUCE,
          0.1,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'dataset_reduce_noclip_num_examples',
          loop_builder.LoopImplementation.DATASET_REDUCE,
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
      (
          'dataset_iterator_noclip_uniform',
          loop_builder.LoopImplementation.DATASET_ITERATOR,
          0.1,
          client_weight_lib.ClientWeighting.UNIFORM,
      ),
      (
          'dataset_iterator_noclip_num_examples',
          loop_builder.LoopImplementation.DATASET_ITERATOR,
          0.1,
          client_weight_lib.ClientWeighting.NUM_EXAMPLES,
      ),
  )
  def test_client_tf(self, loop_implementation, expected_norm, weighting):
    client_tf = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=weighting,
            loop_implementation=loop_implementation,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    dataset = create_test_dataset()
    client_result, model_output = self.evaluate(
        client_tf(optimizer, create_test_initial_weights(), dataset)
    )
    # Both trainable parameters should have been updated, and we don't return
    # the non-trainable variable.
    for trainable_param in client_result.update:
      self.assertAllGreater(np.linalg.norm(trainable_param), expected_norm)
    if weighting == client_weight_lib.ClientWeighting.UNIFORM:
      self.assertEqual(client_result.update_weight, 1.0)
    else:
      self.assertEqual(client_result.update_weight, 12.0)
    expected = {'num_examples': 12}
    self.assertEqual(model_output, {**model_output, **expected})
    self.assertBetween(model_output['loss'][0], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    dataset = create_test_dataset()
    init_weights = create_test_initial_weights()
    init_weights.trainable[1] = bad_value
    client_outputs = client_tf(optimizer, init_weights, dataset)
    # Since this client has non-finite model weights update, we will zero out
    # its `update` and set `update_weight` to zero.
    self.assertEqual(client_outputs[0].update_weight, 0.0)
    self.assertAllClose(client_outputs[0].update, [[[0.0], [0.0]], 0.0])
    # Since this client has non-finite model weights update, we will reset its
    # local metrics, which essentially excludes it from server aggregation.
    tf.nest.map_structure(
        self.assertAllEqual,
        client_outputs[1],
        collections.OrderedDict(loss=[0.0, 0.0], num_examples=0),
    )

  def test_correct_update_weight_with_traced_function(self):
    client_tf = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    optimizer = sgdm.build_sgdm(learning_rate=0.1)
    init_weights = create_test_initial_weights()
    dataset_with_nan = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, np.nan]],
            y=[[0.0], [0.0], [1.0], [1.0]],
        )
    ).batch(1)
    dataset_wo_nan = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            y=[[0.0], [0.0], [1.0], [1.0]],
        )
    ).batch(1)
    # Obtain a concrete function after federated_language.framework.
    client_concrete_fn = client_tf.get_concrete_function(
        optimizer, init_weights, dataset_wo_nan, optimizer_hparams=None
    )
    # Execute the traced function with different data inputs. If the data input
    # has nan, the corresponding model delta will be non-finite, and hence,
    # the `update_weight` should be 0.0 (i.e., exclude this client from model
    # aggregation). Otherwise, the `update_weight` should be larger than 0.0.
    client_outputs = client_concrete_fn(
        optimizer, init_weights, dataset_wo_nan, optimizer_hparams=None
    )
    self.assertGreater(client_outputs[0].update_weight, 0.0)
    client_outputs = client_concrete_fn(
        optimizer, init_weights, dataset_with_nan, optimizer_hparams=None
    )
    self.assertEqual(client_outputs[0].update_weight, 0.0)

  def test_custom_metrics_aggregator(self):

    def sum_then_finalize_then_times_two(metric_finalizers):

      @federated_language.federated_computation
      def aggregation_computation(client_local_unfinalized_metrics):
        unfinalized_metrics_sum = federated_language.federated_sum(
            client_local_unfinalized_metrics
        )

        @tensorflow_computation.tf_computation
        def finalizer_computation(unfinalized_metrics):
          finalized_metrics = collections.OrderedDict()
          for metric_name, metric_finalizer in metric_finalizers.items():
            finalized_metrics[metric_name] = (
                metric_finalizer(unfinalized_metrics[metric_name]) * 2
            )
          return finalized_metrics

        return federated_language.federated_map(
            finalizer_computation, unfinalized_metrics_sum
        )

      return aggregation_computation

    process = model_delta_client_work.build_model_delta_client_work(
        model_fn=create_model,
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        metrics_aggregator=sum_then_finalize_then_times_two,
    )
    client_model_weights = [create_test_initial_weights()]
    client_data = [create_test_dataset()]
    output = process.next(
        process.initialize(), client_model_weights, client_data
    )
    # Train metrics should be multiplied by two by the custom aggregator.
    self.assertEqual(output.measurements['train']['num_examples'], 24)

  @parameterized.named_parameters(
      ('dataset_reduce', loop_builder.LoopImplementation.DATASET_REDUCE),
      ('dataset_iterator', loop_builder.LoopImplementation.DATASET_ITERATOR),
  )
  @mock.patch.object(loop_builder, 'build_training_loop')
  def test_client_tf_dataset_loops(self, loop_implementation, mock_method):
    model_delta_client_work.build_model_delta_update_with_tff_optimizer(
        model_fn=create_model,
        weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
        loop_implementation=loop_implementation,
    )
    mock_method.assert_called_once_with(loop_implementation=loop_implementation)

  @parameterized.named_parameters(
      ('tff_simple', sgdm.build_sgdm(learning_rate=1.0)),
      ('tff_momentum', sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)),
  )
  def test_execution_with_optimizer(self, optimizer):
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    client_data = [create_test_dataset()]
    client_model_weights = [create_test_initial_weights()]

    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)

    self.assertCountEqual(output.measurements.keys(), ['train'])

  def test_get_hparams_returns_expected_result_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    state = client_work_process.initialize()

    hparams = client_work_process.get_hparams(state)

    expected_hparams = collections.OrderedDict(learning_rate=1.0, momentum=0.9)
    self.assertDictEqual(hparams, expected_hparams)

  def test_set_hparams_returns_expected_result_with_tff_optimizer(self):
    optimizer = sgdm.build_sgdm(learning_rate=1.0, momentum=0.9)
    client_work_process = model_delta_client_work.build_model_delta_client_work(
        create_model,
        optimizer,
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    state = client_work_process.initialize()
    hparams = collections.OrderedDict(learning_rate=0.5, momentum=0.3)

    state = client_work_process.set_hparams(state, hparams)

    self.assertDictEqual(state, hparams)

  @parameterized.named_parameters(
      ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
      ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
  )
  def test_tff_client_work_uses_optimizer_hparams(self, weighting):
    dataset = create_test_dataset()
    optimizer1 = sgdm.build_sgdm(learning_rate=0.0)
    optimizer2 = sgdm.build_sgdm(learning_rate=0.1)
    optimizer_hparams = collections.OrderedDict(learning_rate=1.0)

    client_update_tff = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=weighting,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    result1 = client_update_tff(
        optimizer1, create_test_initial_weights(), dataset, optimizer_hparams
    )
    client_update_tff = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=create_model,
            weighting=weighting,
            loop_implementation=loop_builder.LoopImplementation.DATASET_REDUCE,
        )
    )
    result2 = client_update_tff(
        optimizer2, create_test_initial_weights(), dataset, optimizer_hparams
    )

    self.assertAllClose(result1[0].update, result2[0].update)
    self.assertEqual(result1[0].update_weight, result2[0].update_weight)
    self.assertAllClose(result1[1], result2[1])


class FunctionalModelDeltaClientWorkExecutionTest(
    tf.test.TestCase, parameterized.TestCase
):

  @parameterized.named_parameters(
      (
          f'{weighting[0]}_{is_simulation_loop[0]}',
          weighting[1],
          is_simulation_loop[1],
      )
      for weighting, is_simulation_loop in itertools.product(
          [
              ('uniform', client_weight_lib.ClientWeighting.UNIFORM),
              ('num_examples', client_weight_lib.ClientWeighting.NUM_EXAMPLES),
          ],
          [
              (
                  'dataset_iterator',
                  loop_builder.LoopImplementation.DATASET_ITERATOR,
              ),
              (
                  'dataset_reduce',
                  loop_builder.LoopImplementation.DATASET_REDUCE,
              ),
          ],
      )
  )
  def test_functional_model_matches_model_fn(
      self, weighting, loop_implementation
  ):
    dataset = create_test_dataset()

    # Build a FunctionalModel based client_model_update procedure. This will
    # be compared to a model_fn based implementation built below.
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2
    )
    loss_fn = tf.keras.losses.MeanSquaredError()
    input_spec = dataset.element_spec
    functional_model = functional.functional_model_from_keras(
        keras_model, loss_fn=loss_fn, input_spec=input_spec
    )

    # Note: we must wrap in a `tf_computation` for the correct graph-context
    # processing of Keras models wrapped as FunctionalModel.
    @tensorflow_computation.tf_computation
    def client_update_functional_model(model_weights, dataset):
      model_delta_fn = (
          model_delta_client_work.build_functional_model_delta_update(
              model=functional_model,
              weighting=weighting,
              loop_implementation=loop_implementation,
          )
      )
      return model_delta_fn(
          sgdm.build_sgdm(learning_rate=0.1), model_weights, dataset
      )

    # Build a model_fn based client_model_update procedure. This will be
    # comapred to the FunctionalModel variant built above to ensure they
    # procduce the same results.
    def model_fn():
      keras_model = (
          model_examples.build_linear_regression_keras_functional_model(
              feature_dims=2
          )
      )
      loss_fn = tf.keras.losses.MeanSquaredError()
      input_spec = dataset.element_spec
      return keras_utils.from_keras_model(
          keras_model, loss=loss_fn, input_spec=input_spec
      )

    client_update_model_fn = (
        model_delta_client_work.build_model_delta_update_with_tff_optimizer(
            model_fn=model_fn,
            weighting=weighting,
            loop_implementation=loop_implementation,
        )
    )
    model_fn_optimizer = sgdm.build_sgdm(learning_rate=0.1)
    model_fn_weights = model_weights_lib.ModelWeights.from_model(model_fn())

    functional_model_weights = functional_model.initial_weights
    for _ in range(10):
      # pylint: disable=cell-var-from-loop
      model_fn_output, _ = client_update_model_fn(
          model_fn_optimizer, model_fn_weights, dataset
      )
      functional_model_output, _ = client_update_functional_model(
          functional_model_weights, dataset
      )
      self.assertAllClose(
          model_fn_output.update, functional_model_output.update
      )
      self.assertAllClose(
          model_fn_output.update_weight, functional_model_output.update_weight
      )
      model_fn_weights = model_weights_lib.ModelWeights(
          trainable=tf.nest.map_structure(
              lambda u, v: u + v * model_fn_output.update_weight,
              model_fn_weights.trainable,
              model_fn_output.update,
          ),
          non_trainable=model_fn_weights.non_trainable,
      )
      functional_model_weights = (
          tf.nest.map_structure(
              lambda u, v: u + v * functional_model_output.update_weight,
              functional_model_weights[0],
              functional_model_output.update,
          ),
          functional_model_weights[1],
      )
      # pylint: enable=cell-var-from-loop
    self.assertAllClose(tuple(model_fn_weights), functional_model_weights)

  def _create_dataset_and_functional_model(self):
    keras_model = model_examples.build_linear_regression_keras_functional_model(
        feature_dims=2
    )
    loss_fn = tf.keras.losses.MeanSquaredError()
    dataset = create_test_dataset()
    input_spec = dataset.element_spec

    def build_metrics_fn():
      return collections.OrderedDict(num_examples=counters.NumExamplesCounter())

    functional_model = functional.functional_model_from_keras(
        keras_model,
        loss_fn=loss_fn,
        input_spec=input_spec,
        metrics_constructor=build_metrics_fn,
    )
    return dataset, functional_model

  def test_metrics_output(self):
    dataset, functional_model = self._create_dataset_and_functional_model()
    process = model_delta_client_work.build_functional_model_delta_client_work(
        model=functional_model,
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    num_clients = 3
    client_model_weights = [functional_model.initial_weights] * num_clients
    client_datasets = [dataset] * num_clients
    output = process.next(
        process.initialize(), client_model_weights, client_datasets
    )
    self.assertEqual(
        output.measurements['train']['num_examples'], 12 * num_clients
    )

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_metrics_output_with_non_finite_updates(self, bad_value):
    dataset, functional_model = self._create_dataset_and_functional_model()
    process = model_delta_client_work.build_functional_model_delta_client_work(
        model=functional_model,
        optimizer=sgdm.build_sgdm(learning_rate=1.0),
        client_weighting=client_weight_lib.ClientWeighting.NUM_EXAMPLES,
    )
    initial_weights = (
        (
            np.array([[0.0], [0.0]], dtype=np.float32),
            np.array([bad_value], dtype=np.float32),
        ),
        (),
    )
    num_clients = 3
    client_model_weights = [initial_weights] * num_clients
    client_datasets = [dataset] * num_clients
    output = process.next(
        process.initialize(), client_model_weights, client_datasets
    )
    # Since all clients have non-finite model weights update, we reset their
    # local metrics. The resulting aggregated metric at server is zero.
    self.assertEqual(output.measurements['train']['num_examples'], 0)


if __name__ == '__main__':
  execution_contexts.set_sync_local_cpp_execution_context()
  tf.test.main()
