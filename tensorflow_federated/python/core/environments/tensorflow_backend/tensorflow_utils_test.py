# Copyright 2018, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections

from absl.testing import absltest
import attrs
import federated_language
from federated_language.proto import computation_pb2 as pb
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_test_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import tensorflow_utils
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types


class GraphUtilsTest(tf.test.TestCase):

  def _assert_binding_matches_type_and_value(
      self, binding, type_spec, val, graph, is_output
  ):
    """Asserts that 'bindings' matches the given type, value, and graph."""
    self.assertIsInstance(binding, pb.TensorFlow.Binding)
    self.assertIsInstance(type_spec, federated_language.Type)
    binding_oneof = binding.WhichOneof('binding')
    if binding_oneof == 'tensor':
      self.assertTrue(tf.is_tensor(val))
      if is_output:
        # Output tensor names must not match, because `val` might also be in the
        # input binding, causing the same tensor to appear in the `feeds` and
        # `fetches` of the `Session.run()` wich is disallowed by TensorFlow.
        self.assertNotEqual(binding.tensor.tensor_name, val.name)
      else:
        # Input binding names are expected to match
        self.assertEqual(binding.tensor.tensor_name, val.name)
      self.assertIsInstance(type_spec, federated_language.TensorType)
      self.assertEqual(type_spec.dtype, val.dtype.base_dtype)
      self.assertEqual(type_spec.shape, val.shape)
    elif binding_oneof == 'sequence':
      self.assertIsInstance(val, tf.data.Dataset)
      sequence_oneof = binding.sequence.WhichOneof('binding')
      self.assertEqual(sequence_oneof, 'variant_tensor_name')
      variant_tensor = graph.get_tensor_by_name(
          binding.sequence.variant_tensor_name
      )
      op = str(variant_tensor.op.type)
      self.assertTrue((op == 'Placeholder') or (op == 'Identity'))
      self.assertEqual(variant_tensor.dtype, tf.variant)
      self.assertIsInstance(type_spec, federated_language.SequenceType)
      self.assertEqual(
          tensorflow_types.to_type(val.element_spec),
          type_spec.element,
      )
    elif binding_oneof == 'struct':
      self.assertIsInstance(type_spec, federated_language.StructType)
      if not isinstance(val, (list, tuple, structure.Struct)):
        self.assertIsInstance(val, dict)
        val = list(val.values())
      for idx, e in enumerate(type_spec.items()):
        self._assert_binding_matches_type_and_value(
            binding.struct.element[idx], e[1], val[idx], graph, is_output
        )
    else:
      self.fail('Unknown binding.')

  def _assert_input_binding_matches_type_and_value(
      self, binding, type_spec, val, graph
  ):
    self._assert_binding_matches_type_and_value(
        binding, type_spec, val, graph, is_output=False
    )

  def _assert_output_binding_matches_type_and_value(
      self, binding, type_spec, val, graph
  ):
    self._assert_binding_matches_type_and_value(
        binding, type_spec, val, graph, is_output=True
    )

  def _assert_captured_result_eq_dtype(self, type_spec, binding, dtype):
    self.assertIsInstance(type_spec, federated_language.TensorType)
    self.assertEqual(str(type_spec), dtype)
    self.assertEqual(binding.WhichOneof('binding'), 'tensor')

  def _assert_is_placeholder(self, x, name, dtype, shape, graph):
    """Verifies that 'x' is a placeholder with the given attributes."""
    self.assertEqual(x.name, name)
    self.assertEqual(x.dtype, dtype)
    expected_rank = len(shape)
    self.assertEqual(x.shape.rank, expected_rank)
    for i, s in enumerate(shape):
      self.assertEqual(x.shape.dims[i].value, s)
    self.assertEqual(x.op.type, 'Placeholder')
    self.assertIs(x.graph, graph)

  def _checked_capture_result(self, result):
    """Returns the captured result type after first verifying the binding."""
    graph = tf.compat.v1.get_default_graph()
    type_spec, binding = tensorflow_utils.capture_result_from_graph(
        result, graph
    )
    # If the input is a tensor (but not a tf.Variable), ensure that an identity
    # operation was added.
    if tf.is_tensor(result) and not hasattr(result, 'read_value'):
      self.assertNotEqual(result.name, binding.tensor.tensor_name)
    self._assert_output_binding_matches_type_and_value(
        binding, type_spec, result, graph
    )
    return type_spec

  def _checked_stamp_parameter(self, name, spec, graph=None):
    """Returns object stamped in the graph after verifying its bindings."""
    if graph is None:
      graph = tf.compat.v1.get_default_graph()
    val, binding = tensorflow_utils.stamp_parameter_in_graph(name, spec, graph)
    self._assert_input_binding_matches_type_and_value(
        binding, tensorflow_types.to_type(spec), val, graph
    )
    return val

  def test_stamp_parameter_in_graph_with_scalar_int_explicit_graph(self):
    my_graph = tf.Graph()
    x = self._checked_stamp_parameter('foo', tf.int32, my_graph)
    self._assert_is_placeholder(x, 'foo:0', tf.int32, [], my_graph)

  def test_stamp_parameter_in_graph_with_int_vector_implicit_graph(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('bar', (tf.int32, [5]))
    self._assert_is_placeholder(x, 'bar:0', tf.int32, [5], my_graph)

  def test_stamp_parameter_in_graph_with_int_vector_undefined_size(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter('bar', (tf.int32, [None]))
    self._assert_is_placeholder(x, 'bar:0', tf.int32, [None], my_graph)

  def test_stamp_parameter_in_graph_with_struct(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter(
          'foo',
          federated_language.StructType([
              ('a', np.int32),
              ('b', np.bool_),
          ]),
      )
    self.assertIsInstance(x, structure.Struct)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_struct_with_python_type(self):
    with tf.Graph().as_default() as my_graph:
      x = self._checked_stamp_parameter(
          'foo',
          federated_language.StructWithPythonType(
              [
                  ('a', np.int32),
                  ('b', np.bool_),
              ],
              collections.OrderedDict,
          ),
      )
    self.assertIsInstance(x, structure.Struct)
    self.assertTrue(len(x), 2)
    self._assert_is_placeholder(x.a, 'foo_a:0', tf.int32, [], my_graph)
    self._assert_is_placeholder(x.b, 'foo_b:0', tf.bool, [], my_graph)

  def test_stamp_parameter_in_graph_with_bool_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', federated_language.SequenceType(np.bool_)
      )
      self.assertIsInstance(x, tf.data.Dataset)
      self.assertEqual(x.element_spec, tf.TensorSpec(shape=(), dtype=tf.bool))

  def test_stamp_parameter_in_graph_with_int_vector_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo', federated_language.SequenceType((np.int32, [50]))
      )
      self.assertIsInstance(x, tf.data.Dataset)
      self.assertEqual(
          x.element_spec, tf.TensorSpec(shape=(50,), dtype=tf.int32)
      )

  def test_stamp_parameter_in_graph_with_tensor_ordered_dict_sequence(self):
    with tf.Graph().as_default():
      x = self._checked_stamp_parameter(
          'foo',
          federated_language.SequenceType(
              collections.OrderedDict(
                  [('A', (np.float32, [3, 4, 5])), ('B', (np.int32, [1]))]
              )
          ),
      )
      self.assertIsInstance(x, tf.data.Dataset)
      self.assertEqual(
          x.element_spec,
          {
              'A': tf.TensorSpec(shape=(3, 4, 5), dtype=tf.float32),
              'B': tf.TensorSpec(shape=(1,), dtype=tf.int32),
          },
      )

  def test_capture_result_with_string(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          'a', graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'str')

  def test_capture_result_with_int(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(1, graph)
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32')

  def test_capture_result_with_float(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          1.0, graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float32')

  def test_capture_result_with_bool(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          True, graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'bool')

  def test_capture_result_with_np_int32(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.int32(1), graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32')

  def test_capture_result_with_np_int64(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.int64(1), graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int64')

  def test_capture_result_with_np_float32(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.float32(1.0), graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float32')

  def test_capture_result_with_np_float64(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.float64(1.0), graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'float64')

  def test_capture_result_with_np_bool(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.bool_(True), graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'bool')

  def test_capture_result_with_np_ndarray(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          np.ndarray(shape=(2, 0), dtype=np.int32), graph
      )
    self._assert_captured_result_eq_dtype(type_spec, binding, 'int32[2,0]')

  def test_capture_result_with_ragged_tensor(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          tf.RaggedTensor.from_row_splits([0, 0, 0, 0], [0, 1, 4]), graph
      )
      del binding
      self.assertEqual(
          type_spec,
          federated_language.StructWithPythonType(
              [
                  ('flat_values', federated_language.TensorType(np.int32, [4])),
                  (
                      'nested_row_splits',
                      federated_language.StructWithPythonType(
                          [(
                              None,
                              federated_language.TensorType(np.int64, [3]),
                          )],
                          tuple,
                      ),
                  ),
              ],
              tf.RaggedTensor,
          ),
      )

  def test_capture_result_with_sparse_tensor(self):
    with tf.Graph().as_default() as graph:
      type_spec, binding = tensorflow_utils.capture_result_from_graph(
          tf.SparseTensor(indices=[[1]], values=[2], dense_shape=[5]), graph
      )
      del binding
      self.assertEqual(
          type_spec,
          federated_language.StructWithPythonType(
              [
                  ('indices', federated_language.TensorType(np.int64, [1, 1])),
                  ('values', federated_language.TensorType(np.int32, [1])),
                  ('dense_shape', federated_language.TensorType(np.int64, [1])),
              ],
              tf.SparseTensor,
          ),
      )

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_int_placeholder(self):
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.compat.v1.placeholder(tf.int32, shape=[])
            )
        ),
        'int32',
    )

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_int_variable(self):
    # Verifies that the variable dtype is not being captured as `int32_ref`,
    # since TFF has no concept of passing arguments by reference.
    self.assertEqual(
        str(
            self._checked_capture_result(
                tf.Variable(
                    initial_value=0, name='foo', dtype=tf.int32, shape=[]
                )
            )
        ),
        'int32',
    )

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_list_of_constants(self):
    t = self._checked_capture_result([tf.constant(1), tf.constant(True)])
    self.assertEqual(str(t), '<int32,bool>')
    self.assertIs(t.python_container, list)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_tuple_of_constants(self):
    t = self._checked_capture_result((tf.constant(1), tf.constant(True)))
    self.assertEqual(str(t), '<int32,bool>')
    self.assertIs(t.python_container, tuple)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_dict_of_constants(self):
    t1 = self._checked_capture_result({
        'a': tf.constant(1),
        'b': tf.constant(True),
    })
    self.assertEqual(str(t1), '<a=int32,b=bool>')
    self.assertIs(t1.python_container, dict)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_ordered_dict_of_constants(self):
    t = self._checked_capture_result(
        collections.OrderedDict([
            ('b', tf.constant(True)),
            ('a', tf.constant(1)),
        ])
    )
    self.assertEqual(str(t), '<b=bool,a=int32>')
    self.assertIs(t.python_container, collections.OrderedDict)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_ordered_dict_with_non_string_keys_throws(self):
    value = collections.OrderedDict([(1, 2)])
    graph = tf.compat.v1.get_default_graph()
    with self.assertRaises(tensorflow_utils.DictionaryKeyMustBeStringError):
      tensorflow_utils.capture_result_from_graph(value, graph)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_unknown_class_throws(self):
    class UnknownClass:
      pass

    value = UnknownClass()
    graph = tf.compat.v1.get_default_graph()
    with self.assertRaises(tensorflow_utils.UnsupportedGraphResultError):
      tensorflow_utils.capture_result_from_graph(value, graph)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_namedtuple_of_constants(self):
    test_named_tuple = collections.namedtuple('_', 'x y')
    t = self._checked_capture_result(
        test_named_tuple(tf.constant(1), tf.constant(True))
    )
    self.assertEqual(str(t), '<x=int32,y=bool>')
    self.assertIs(t.python_container, test_named_tuple)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_attrs_of_constants(self):

    @attrs.define
    class TestFoo:
      x: int
      y: bool

    graph = tf.compat.v1.get_default_graph()
    type_spec, _ = tensorflow_utils.capture_result_from_graph(
        TestFoo(tf.constant(1), tf.constant(True)), graph
    )
    self.assertEqual(str(type_spec), '<x=int32,y=bool>')
    self.assertIs(type_spec.python_container, TestFoo)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_struct_of_constants(self):
    t = self._checked_capture_result(
        structure.Struct([
            ('x', tf.constant(10)),
            (None, tf.constant(True)),
            ('y', tf.constant(0.66)),
        ])
    )
    self.assertEqual(str(t), '<x=int32,bool,y=float32>')
    self.assertIsInstance(t, federated_language.StructType)
    self.assertNotIsInstance(t, federated_language.StructWithPythonType)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_nested_lists_and_tuples(self):
    named_tuple_type = collections.namedtuple('_', 'a b')
    t = self._checked_capture_result(
        structure.Struct([
            (
                'x',
                named_tuple_type(
                    {'p': {'q': tf.constant(True)}}, [tf.constant(False)]
                ),
            ),
            (None, [[tf.constant(10)]]),
        ])
    )
    self.assertEqual(str(t), '<x=<a=<p=<q=bool>>,b=<bool>>,<<int32>>>')
    self.assertIsInstance(t, federated_language.StructType)
    self.assertNotIsInstance(t, federated_language.StructWithPythonType)
    self.assertIsInstance(t.x, federated_language.StructWithPythonType)
    self.assertIs(t.x.python_container, named_tuple_type)
    self.assertIsInstance(t[1], federated_language.StructWithPythonType)
    self.assertIs(t[1].python_container, list)

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_sequence_of_ints_using_from_tensors(self):
    ds = tf.data.Dataset.from_tensors(tf.constant(10))
    self.assertEqual(str(self._checked_capture_result(ds)), 'int32*')

  @tensorflow_test_utils.graph_mode_test
  def test_capture_result_with_sequence_of_ints_using_range(self):
    ds = tf.data.Dataset.range(10)
    self.assertEqual(str(self._checked_capture_result(ds)), 'int64*')

  def test_compute_map_from_bindings_with_tuple_of_tensors(self):
    with tf.Graph().as_default() as graph:
      _, source = tensorflow_utils.capture_result_from_graph(
          collections.OrderedDict([
              ('foo', tf.constant(10, name='A')),
              ('bar', tf.constant(20, name='B')),
          ]),
          graph,
      )
      _, target = tensorflow_utils.capture_result_from_graph(
          collections.OrderedDict([
              ('foo', tf.constant(30, name='C')),
              ('bar', tf.constant(40, name='D')),
          ]),
          graph,
      )
    result = tensorflow_utils._compute_map_from_bindings(source, target)
    self.assertAllEqual(
        result,
        collections.OrderedDict(
            [('Identity:0', 'Identity_2:0'), ('Identity_1:0', 'Identity_3:0')]
        ),
    )

  def test_compute_map_from_bindings_with_sequence(self):
    source = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo')
    )
    target = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='bar')
    )
    result = tensorflow_utils._compute_map_from_bindings(source, target)
    self.assertEqual(result, collections.OrderedDict(foo='bar'))

  def test_extract_tensor_names_from_binding_with_tuple_of_tensors(self):
    with tf.Graph().as_default() as graph:
      _, binding = tensorflow_utils.capture_result_from_graph(
          collections.OrderedDict([
              ('foo', tf.constant(10, name='A')),
              ('bar', tf.constant(20, name='B')),
          ]),
          graph,
      )
    result = tensorflow_utils.extract_tensor_names_from_binding(binding)
    self.assertEqual(result, ['Identity:0', 'Identity_1:0'])

  def test_extract_tensor_names_from_binding_with_sequence(self):
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo')
    )
    result = tensorflow_utils.extract_tensor_names_from_binding(binding)
    self.assertEqual(str(result), "['foo']")

  @tensorflow_test_utils.graph_mode_test
  def test_assemble_result_from_graph_with_named_tuple(self):
    test_named_tuple = collections.namedtuple('_', 'X Y')
    type_spec = test_named_tuple(tf.int32, tf.int32)
    binding = pb.TensorFlow.Binding(
        struct=pb.TensorFlow.StructBinding(
            element=[
                pb.TensorFlow.Binding(
                    tensor=pb.TensorFlow.TensorBinding(tensor_name='P')
                ),
                pb.TensorFlow.Binding(
                    tensor=pb.TensorFlow.TensorBinding(tensor_name='Q')
                ),
            ]
        )
    )
    tensor_a = tf.constant(1, name='A')
    tensor_b = tf.constant(2, name='B')
    output_map = {'P': tensor_a, 'Q': tensor_b}
    result = tensorflow_utils._assemble_result_from_graph(
        type_spec, binding, output_map
    )
    self.assertIsInstance(result, test_named_tuple)
    self.assertEqual(result.X, tensor_a)
    self.assertEqual(result.Y, tensor_b)

  @tensorflow_test_utils.graph_mode_test
  def test_assemble_result_from_graph_with_sequence_of_odicts(self):
    type_spec = federated_language.SequenceType(
        collections.OrderedDict([('X', np.int32), ('Y', np.int32)])
    )
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo')
    )
    data_set = tf.data.Dataset.from_tensors(
        {'X': tf.constant(1), 'Y': tf.constant(2)}
    )
    output_map = {'foo': tf.data.experimental.to_variant(data_set)}
    result = tensorflow_utils._assemble_result_from_graph(
        type_spec, binding, output_map
    )
    self.assertIsInstance(result, tf.data.Dataset)
    self.assertEqual(
        result.element_spec,
        collections.OrderedDict([
            ('X', tf.TensorSpec(shape=(), dtype=tf.int32)),
            ('Y', tf.TensorSpec(shape=(), dtype=tf.int32)),
        ]),
    )

  @tensorflow_test_utils.graph_mode_test
  def test_assemble_result_from_graph_with_sequence_of_namedtuples(self):
    named_tuple_type = collections.namedtuple('TestNamedTuple', 'X Y')
    type_spec = federated_language.SequenceType(
        named_tuple_type(np.int32, np.int32)
    )
    binding = pb.TensorFlow.Binding(
        sequence=pb.TensorFlow.SequenceBinding(variant_tensor_name='foo')
    )
    data_set = tf.data.Dataset.from_tensors(
        {'X': tf.constant(1), 'Y': tf.constant(2)}
    )
    output_map = {'foo': tf.data.experimental.to_variant(data_set)}
    result = tensorflow_utils._assemble_result_from_graph(
        type_spec, binding, output_map
    )
    self.assertIsInstance(result, tf.data.Dataset)
    self.assertEqual(
        result.element_spec,
        named_tuple_type(
            X=tf.TensorSpec(shape=(), dtype=tf.int32),
            Y=tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

  def test__make_whimsy_element_for_type_spec_raises_sequence_type(self):
    type_spec = federated_language.SequenceType(np.float32)
    with self.assertRaisesRegex(
        ValueError, 'Cannot construct array for TFF type'
    ):
      tensorflow_utils._make_whimsy_element_for_type_spec(type_spec)

  def test__make_whimsy_element_for_type_spec_raises_negative_none_dim_replacement(
      self,
  ):
    with self.assertRaisesRegex(ValueError, 'nonnegative'):
      tensorflow_utils._make_whimsy_element_for_type_spec(tf.float32, -1)

  def test_make_whimsy_element_tensor_type(self):
    type_spec = federated_language.TensorType(
        np.float32, [None, 10, None, 10, 10]
    )
    elem = tensorflow_utils._make_whimsy_element_for_type_spec(type_spec)
    correct_elem = np.zeros([0, 10, 0, 10, 10], np.float32)
    self.assertAllClose(elem, correct_elem)

  def test_make_whimsy_element_tensor_type_none_replaced_by_1(self):
    type_spec = federated_language.TensorType(
        np.float32, [None, 10, None, 10, 10]
    )
    elem = tensorflow_utils._make_whimsy_element_for_type_spec(
        type_spec, none_dim_replacement=1
    )
    correct_elem = np.zeros([1, 10, 1, 10, 10], np.float32)
    self.assertAllClose(elem, correct_elem)

  def test_make_whimsy_element_struct_type(self):
    tensor1 = federated_language.TensorType(
        np.float32, [None, 10, None, 10, 10]
    )
    tensor2 = federated_language.TensorType(np.int32, [10, None, 10])
    namedtuple = federated_language.StructType([('x', tensor1), ('y', tensor2)])
    unnamedtuple = federated_language.StructType(
        [('x', tensor1), ('y', tensor2)]
    )
    elem = tensorflow_utils._make_whimsy_element_for_type_spec(namedtuple)
    correct_list = [
        np.zeros([0, 10, 0, 10, 10], np.float32),
        np.zeros([10, 0, 10], np.int32),
    ]
    self.assertEqual(len(elem), len(correct_list))
    for k, _ in enumerate(elem):
      self.assertAllClose(elem[k], correct_list[k])
    unnamed_elem = tensorflow_utils._make_whimsy_element_for_type_spec(
        unnamedtuple
    )
    self.assertEqual(len(unnamed_elem), len(correct_list))
    for k, _ in enumerate(unnamed_elem):
      self.assertAllClose(unnamed_elem[k], correct_list[k])

  def test_make_data_set_from_elements_in_eager_context(self):
    ds = tensorflow_utils.make_data_set_from_elements(None, [10, 20], tf.int32)
    self.assertCountEqual([x.numpy() for x in iter(ds)], [10, 20])

  def test_make_data_set_from_sparse_tensor_elements(self):
    self.skipTest('b/258037897')
    sparse_tensor = tf.SparseTensor(
        [[0, i] for i in range(5)], list(range(5)), [1, 10]
    )
    ds = tf.data.Dataset.from_tensor_slices(sparse_tensor)
    constructed_ds = tensorflow_utils.make_data_set_from_elements(
        None, list(ds), tensorflow_types.to_type(ds.element_spec)
    )
    self.assertEqual(ds.element_spec, constructed_ds.element_spec)
    self.assertEqual(
        tf.sparse.to_dense(next(iter(ds))),
        tf.sparse.to_dense(next(iter(constructed_ds))),
    )

  def test_make_data_set_from_ragged_tensor_elements(self):
    self.skipTest('b/258038191')
    ragged_tensor = tf.RaggedTensor.from_row_splits([0, 0, 0, 0], [0, 1, 4])
    constructed_ds = tensorflow_utils.make_data_set_from_elements(
        None,
        [ragged_tensor],
        tensorflow_types.to_type(tf.RaggedTensorSpec.from_value(ragged_tensor)),
    )
    self.assertIsInstance(constructed_ds.element_spec, tf.RaggedTensorSpec)
    self.assertEqual(
        next(iter(constructed_ds)).to_tensor(), ragged_tensor.to_tensor()
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [], tf.float32
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.compat.v1.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list_definite_tensor(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [],
        federated_language.TensorType(np.float32, [None, 10]),
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        ds.element_spec, tf.TensorSpec(shape=(0, 10), dtype=tf.float32)
    )
    self.assertEqual(
        tf.compat.v1.Session().run(ds.reduce(1.0, lambda x, y: x + y)), 1.0
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_empty_list_definite_tuple(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [],
        [
            federated_language.TensorType(np.float32, [None, 10]),
            federated_language.TensorType(np.float32, [None, 5]),
        ],
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        ds.element_spec,
        (
            tf.TensorSpec(shape=(0, 10), dtype=tf.float32),
            tf.TensorSpec(shape=(0, 5), dtype=tf.float32),
        ),
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_ints(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(), [1, 2, 3, 4], tf.int32
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.compat.v1.Session().run(ds.reduce(0, lambda x, y: x + y)), 10
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            {
                'a': 1,
                'b': 2,
            },
            {
                'a': 3,
                'b': 4,
            },
        ],
        [('a', tf.int32), ('b', tf.int32)],
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + y['a'] + y['b'])
        ),
        10,
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_ordered_dicts(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            collections.OrderedDict([
                ('a', 1),
                ('b', 2),
            ]),
            collections.OrderedDict([
                ('a', 3),
                ('b', 4),
            ]),
        ],
        [('a', tf.int32), ('b', tf.int32)],
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + y['a'] + y['b'])
        ),
        10,
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_lists(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            [[1], [2]],
            [[3], [4]],
        ],
        [[tf.int32], [tf.int32]],
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + tf.reduce_sum(y))
        ),
        10,
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_structs(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            structure.Struct([
                ('a', 1),
                ('b', 2),
            ]),
            structure.Struct([
                ('a', 3),
                ('b', 4),
            ]),
        ],
        [('a', tf.int32), ('b', tf.int32)],
    )
    self.assertIsInstance(ds, tf.data.Dataset)
    self.assertEqual(
        tf.compat.v1.Session().run(
            ds.reduce(0, lambda x, y: x + y['a'] + y['b'])
        ),
        10,
    )

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_lists(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            {
                'a': [1],
                'b': [2],
            },
            {
                'a': [3],
                'b': [4],
            },
        ],
        [('a', [tf.int32]), ('b', [tf.int32])],
    )

    self.assertIsInstance(ds, tf.data.Dataset)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.compat.v1.Session().run(ds.reduce(0, reduce_fn)), 10)

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_tensors(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            {
                'a': 1,
                'b': 2,
            },
            {
                'a': 3,
                'b': 4,
            },
        ],
        [('a', tf.int32), ('b', tf.int32)],
    )

    self.assertIsInstance(ds, tf.data.Dataset)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.compat.v1.Session().run(ds.reduce(0, reduce_fn)), 10)

  @tensorflow_test_utils.graph_mode_test
  def test_make_data_set_from_elements_with_list_of_dicts_with_np_array(self):
    ds = tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            {
                'a': np.array([1], dtype=np.int32),
                'b': np.array([2], dtype=np.int32),
            },
            {
                'a': np.array([3], dtype=np.int32),
                'b': np.array([4], dtype=np.int32),
            },
        ],
        [('a', (tf.int32, [1])), ('b', (tf.int32, [1]))],
    )
    self.assertIsInstance(ds, tf.data.Dataset)

    def reduce_fn(x, y):
      return x + tf.reduce_sum(y['a']) + tf.reduce_sum(y['b'])

    self.assertEqual(tf.compat.v1.Session().run(ds.reduce(0, reduce_fn)), 10)

  @tensorflow_test_utils.graph_mode_test
  def test_fetch_value_in_session_with_string(self):
    x = tf.constant('abc')
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), 'abc')

  @tensorflow_test_utils.graph_mode_test
  def test_fetch_value_in_session_without_data_sets(self):
    x = structure.Struct([
        (
            'a',
            structure.Struct([
                ('b', tf.constant(10)),
            ]),
        ),
    ])
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=10>>')

  @tensorflow_test_utils.graph_mode_test
  def test_fetch_value_in_session_with_empty_structure(self):
    x = structure.Struct([
        (
            'a',
            structure.Struct([
                ('b', structure.Struct([])),
            ]),
        ),
    ])
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=<>>>')

  @tensorflow_test_utils.graph_mode_test
  def test_fetch_value_in_session_with_partially_empty_structure(self):
    x = structure.Struct([
        (
            'a',
            structure.Struct([
                ('b', structure.Struct([])),
                ('c', tf.constant(10)),
            ]),
        ),
    ])
    with tf.compat.v1.Session() as sess:
      y = tensorflow_utils.fetch_value_in_session(sess, x)
    self.assertEqual(str(y), '<a=<b=<>,c=10>>')

  def test_make_empty_list_structure_for_element_type_spec_w_tuple_dict(self):
    type_spec = [tf.int32, [('a', tf.bool), ('b', tf.float32)]]
    result = tensorflow_utils._make_empty_list_structure_for_element_type_spec(
        type_spec
    )
    self.assertEqual(result, ([], collections.OrderedDict(a=[], b=[])))

  def test_append_to_list_structure_for_element_type_spec_w_tuple_dict(self):
    nested = tuple([[], collections.OrderedDict([('a', []), ('b', [])])])
    type_spec = [tf.int32, [('a', tf.bool), ('b', tf.float32)]]
    for value in [[10, {'a': True, 'b': 30}], (40, [False, 60])]:
      tensorflow_utils._append_to_list_structure_for_element_type_spec(
          nested, value, type_spec
      )
    self.assertAllEqual(
        nested,
        (
            [tf.constant(10, tf.int32), tf.constant(40, tf.int32)],
            collections.OrderedDict(
                a=[tf.constant(True, tf.bool), tf.constant(False, tf.bool)],
                b=[tf.constant(30, tf.float32), tf.constant(60, tf.float32)],
            ),
        ),
    )

  def test_append_to_list_structure_with_too_few_element_keys(self):
    nested = collections.OrderedDict([('a', []), ('b', [])])
    value = {'a': 10}
    type_spec = [('a', tf.int32), ('b', tf.int32)]
    with self.assertRaises(TypeError):
      tensorflow_utils._append_to_list_structure_for_element_type_spec(
          nested, value, type_spec
      )

  def test_append_to_list_structure_with_too_many_element_keys(self):
    nested = collections.OrderedDict([('a', []), ('b', [])])
    value = {'a': 10, 'b': 20, 'c': 30}
    type_spec = [('a', tf.int32), ('b', tf.int32)]
    with self.assertRaises(TypeError):
      tensorflow_utils._append_to_list_structure_for_element_type_spec(
          nested, value, type_spec
      )

  def test_append_to_list_structure_with_too_few_unnamed_elements(self):
    nested = tuple([[], []])
    value = [10]
    type_spec = [tf.int32, tf.int32]
    with self.assertRaises(TypeError):
      tensorflow_utils._append_to_list_structure_for_element_type_spec(
          nested, value, type_spec
      )

  def test_append_to_list_structure_with_too_many_unnamed_elements(self):
    nested = tuple([[], []])
    value = [10, 20, 30]
    type_spec = [tf.int32, tf.int32]
    with self.assertRaises(TypeError):
      tensorflow_utils._append_to_list_structure_for_element_type_spec(
          nested, value, type_spec
      )

  def test__replace_empty_leaf_lists_with_numpy_arrays(self):
    lists = tuple([[], collections.OrderedDict([('a', []), ('b', [])])])
    type_spec = [tf.int32, [('a', tf.bool), ('b', tf.float32)]]
    result = tensorflow_utils._replace_empty_leaf_lists_with_numpy_arrays(
        lists, type_spec
    )

    expected_structure = tuple([
        np.array([], dtype=np.int32),
        collections.OrderedDict([
            ('a', np.array([], dtype=bool)),
            ('b', np.array([], dtype=np.float32)),
        ]),
    ])

    self.assertEqual(
        str(result).replace(' ', ''), str(expected_structure).replace(' ', '')
    )

  def _test_list_structure(self, type_spec, elements, expected_output):
    result = tensorflow_utils._make_empty_list_structure_for_element_type_spec(
        type_spec
    )
    for element_value in elements:
      tensorflow_utils._append_to_list_structure_for_element_type_spec(
          result, element_value, type_spec
      )
    result = tensorflow_utils._replace_empty_leaf_lists_with_numpy_arrays(
        result, type_spec
    )
    # Use assertAllClose instead of allEqual for the behavior that empty
    # arrays are equal.
    self.assertAllClose(result, expected_output, rtol=0.0, atol=0.0)

  def test_list_structures_from_element_type_spec_with_none_value(self):
    self._test_list_structure(
        [tf.int32, [('a', tf.bool), ('b', tf.float32)]],
        [None],
        (
            np.array([], dtype=np.int32),
            collections.OrderedDict(
                a=np.array([], dtype=bool),
                b=np.array([], dtype=np.float32),
            ),
        ),
    )

  def test_list_structures_from_element_type_spec_with_int_value(self):
    self._test_list_structure(tf.int32, [1], [tf.constant(1, tf.int32)])

  def test_list_structures_from_element_type_spec_with_empty_dict_value(self):
    self._test_list_structure(
        federated_language.StructType([]), [{}], collections.OrderedDict()
    )

  def test_list_structures_from_element_type_spec_with_dict_value(self):
    self._test_list_structure(
        [('a', tf.int32), ('b', tf.int32)],
        [{'a': 1, 'b': 2}, {'a': 1, 'b': 2}],
        collections.OrderedDict(
            a=[
                tf.constant(1, tf.int32),
                tf.constant(1, tf.int32),
            ],
            b=[
                tf.constant(2, tf.int32),
                tf.constant(2, tf.int32),
            ],
        ),
    )

  def test_list_structures_from_element_type_spec_with_no_values(self):
    self._test_list_structure(tf.int32, [], [])

  def test_list_structures_from_element_type_spec_with_int_values(self):
    self._test_list_structure(
        tf.int32,
        [1, 2, 3],
        [
            tf.constant(1, tf.int32),
            tf.constant(2, tf.int32),
            tf.constant(3, tf.int32),
        ],
    )

  def test_list_structures_from_element_type_spec_with_empty_dict_values(self):
    self._test_list_structure(
        federated_language.StructType([]),
        [{}, {}, {}],
        collections.OrderedDict(),
    )

  def test_list_structures_from_element_type_spec_with_structures(self):
    self._test_list_structure(
        federated_language.StructType([('a', np.int32)]),
        [structure.Struct([('a', 1)]), structure.Struct([('a', 2)])],
        collections.OrderedDict(
            a=[
                tf.constant(1, tf.int32),
                tf.constant(2, tf.int32),
            ]
        ),
    )

  def test_list_structures_from_element_type_spec_with_empty_anon_tuples(self):
    self._test_list_structure(
        federated_language.StructType([]),
        [structure.Struct([]), structure.Struct([])],
        collections.OrderedDict(),
    )

  def test_list_structures_from_element_type_spec_w_list_of_anon_tuples(self):
    self._test_list_structure(
        federated_language.StructType(
            [federated_language.StructType([('a', np.int32)])]
        ),
        [[structure.Struct([('a', 1)])], [structure.Struct([('a', 2)])]],
        (
            collections.OrderedDict(
                a=[tf.constant(1, tf.int32), tf.constant(2, tf.int32)],
            ),
        ),
    )

  def test_make_data_set_from_elements_with_wrong_elements(self):
    with self.assertRaises(TypeError):
      tensorflow_utils.make_data_set_from_elements(
          tf.compat.v1.get_default_graph(), [{'a': 1}, {'a': 2}], tf.int32
      )

  def test_make_data_set_from_elements_with_odd_last_batch(self):
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [np.array([1, 2]), np.array([3])],
        federated_language.TensorType(np.int32, (None,)),
    )
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [{'x': np.array([1, 2])}, {'x': np.array([3])}],
        [('x', federated_language.TensorType(np.int32, (None,)))],
    )

  def test_make_data_set_from_elements_with_odd_all_batches(self):
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            np.array([1, 2]),
            np.array([3]),
            np.array([4, 5, 6]),
            np.array([7, 8]),
        ],
        federated_language.TensorType(np.int32, (None,)),
    )
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [
            {'x': np.array([1, 2])},
            {'x': np.array([3])},
            {'x': np.array([4, 5, 6])},
            {'x': np.array([7, 8])},
        ],
        [('x', federated_language.TensorType(np.int32, (None,)))],
    )

  def test_make_data_set_from_elements_with_just_one_batch(self):
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [np.array([1])],
        federated_language.TensorType(np.int32, (None,)),
    )
    tensorflow_utils.make_data_set_from_elements(
        tf.compat.v1.get_default_graph(),
        [{'x': np.array([1])}],
        [('x', federated_language.TensorType(np.int32, (None,)))],
    )

  def test_make_dataset_from_variant_tensor_constructs_dataset(self):
    with tf.Graph().as_default():
      ds = tensorflow_utils._make_dataset_from_variant_tensor(
          tf.data.experimental.to_variant(tf.data.Dataset.range(5)), tf.int64
      )
      self.assertIsInstance(ds, tf.data.Dataset)
      result = ds.reduce(np.int64(0), lambda x, y: x + y)
      with tf.compat.v1.Session() as sess:
        self.assertEqual(sess.run(result), 10)

  def test_make_dataset_from_variant_tensor_fails_with_bad_tensor(self):
    with self.assertRaises(TypeError):
      with tf.Graph().as_default():
        tensorflow_utils._make_dataset_from_variant_tensor(
            tf.constant(10), tf.int32
        )

  def test_make_dataset_from_variant_tensor_fails_with_bad_type(self):
    with self.assertRaises(TypeError):
      with tf.Graph().as_default():
        tensorflow_utils._make_dataset_from_variant_tensor(
            tf.data.experimental.to_variant(tf.data.Dataset.range(5)), 'a'
        )

  def test_coerce_dataset_elements_noop(self):
    x = tf.data.Dataset.range(5)
    y = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        x, federated_language.TensorType(np.int64)
    )
    self.assertEqual(x.element_spec, y.element_spec)

  def test_coerce_ragged_tensor_dataset_elements_noop(self):
    ragged_tensor = tf.RaggedTensor.from_row_splits(
        values=[3, 1, 4], row_splits=[0, 2, 2, 3]
    )
    dataset = tf.data.Dataset.from_tensors(ragged_tensor)
    element_type = federated_language.StructWithPythonType(
        [
            ('flat_values', federated_language.TensorType(np.int32)),
            (
                'nested_row_splits',
                federated_language.StructWithPythonType(
                    [federated_language.TensorType(np.int64, [None])], tuple
                ),
            ),
        ],
        tf.RaggedTensor,
    )
    result = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        dataset, element_type
    )
    self.assertEqual(dataset.element_spec, result.element_spec)

  def test_coerce_sparse_tensor_dataset_elements_noop(self):
    sparse_tensor = tf.sparse.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]
    )
    dataset = tf.data.Dataset.from_tensors(sparse_tensor)
    element_type = federated_language.StructWithPythonType(
        [
            (
                'indices',
                federated_language.TensorType(np.int64, shape=[None, 2]),
            ),
            ('values', federated_language.TensorType(np.int32, shape=[None])),
            ('dense_shape', federated_language.TensorType(np.int64, shape=[2])),
        ],
        tf.sparse.SparseTensor,
    )
    result = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        dataset, element_type
    )
    self.assertEqual(dataset.element_spec, result.element_spec)

  def test_coerce_dataset_elements_nested_structure(self):
    test_tuple_type = collections.namedtuple('TestTuple', ['u', 'v'])

    def _make_nested_tf_structure(x):
      return {
          'b': tf.cast(x, tf.int32),
          'a': tuple([
              x,
              test_tuple_type(x * 2, x * 3),
              collections.OrderedDict([('x', x**2), ('y', x**3)]),
          ]),
          'c': tf.cast(x, tf.float32),
      }

    x = tf.data.Dataset.range(5).map(_make_nested_tf_structure)

    element_type = federated_language.StructType([
        (
            'a',
            federated_language.StructType([
                (None, np.int64),
                (None, test_tuple_type(np.int64, np.int64)),
                (
                    None,
                    federated_language.StructType(
                        [('x', np.int64), ('y', np.int64)]
                    ),
                ),
            ]),
        ),
        ('b', np.int32),
        ('c', np.float32),
    ])

    y = tensorflow_utils.coerce_dataset_elements_to_tff_type_spec(
        x, element_type
    )
    y_type = tensorflow_types.to_type(y.element_spec)
    y_type.check_equivalent_to(element_type)


class TensorFlowDeserializationTest(tf.test.TestCase):

  @tensorflow_test_utils.graph_mode_test
  def test_deserialize_and_call_tf_computation_with_add_one(self):
    with tf.Graph().as_default() as graph:
      parameter_value, parameter_binding = (
          tensorflow_utils.stamp_parameter_in_graph('x', tf.int32, graph)
      )
      result = tf.identity(parameter_value)
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          result, graph
      )
    parameter_type = federated_language.TensorType(np.int32)
    type_signature = federated_language.FunctionType(
        parameter_type, result_type
    )
    tensorflow_proto = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
        parameter=parameter_binding,
        result=result_binding,
    )
    serialized_type = type_signature.to_proto()
    computation_proto = pb.Computation(
        type=serialized_type, tensorflow=tensorflow_proto
    )
    init_op, result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto,
        tf.constant(10),
        tf.compat.v1.get_default_graph(),
        '',
        tf.constant('bogus_token'),
    )
    self.assertTrue(tf.is_tensor(result))
    with tf.compat.v1.Session() as sess:
      if init_op:
        sess.run(init_op)
      result_val = sess.run(result)
    self.assertEqual(result_val, 10)

  @tensorflow_test_utils.graph_mode_test
  def test_deserialize_and_call_tf_computation_with_placeholder_replacement(
      self,
  ):
    with tf.Graph().as_default() as graph:
      parameter_value, parameter_binding = (
          tensorflow_utils.stamp_parameter_in_graph(
              'x', (tf.int32, tf.int32), graph
          )
      )
      # Force a control dependency on a placeholder. This will allow us to
      # assert that control dependencies were also remapped during
      # `tf.graph_util.import_graph_def`.
      with tf.compat.v1.control_dependencies([parameter_value[1]]):
        result = tf.identity(parameter_value[0])
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          result, graph
      )
    parameter_type = federated_language.StructType([
        (None, federated_language.TensorType(np.int32)),
        (None, federated_language.TensorType(np.int32)),
    ])
    type_signature = federated_language.FunctionType(
        parameter_type, result_type
    )
    tensorflow_proto = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
        parameter=parameter_binding,
        result=result_binding,
    )
    serialized_type = type_signature.to_proto()
    computation_proto = pb.Computation(
        type=serialized_type, tensorflow=tensorflow_proto
    )
    init_op, result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto,
        (tf.constant(10), tf.constant(20)),
        tf.compat.v1.get_default_graph(),
        '',
        tf.constant('bogus_token'),
    )
    self.assertTrue(tf.is_tensor(result))
    with tf.compat.v1.Session() as sess:
      if init_op:
        sess.run(init_op)
      result_val = sess.run(result)
    self.assertEqual(result_val, 10)

  @tensorflow_test_utils.graph_mode_test
  def test_deserialize_and_call_tf_computation_returning_session_token(self):
    with tf.Graph().as_default() as graph:
      session_token_placeholder = tf.compat.v1.placeholder(tf.string, shape=())
      result = tf.identity(session_token_placeholder)
      result_type, result_binding = tensorflow_utils.capture_result_from_graph(
          result, graph
      )
    parameter_type = None
    type_signature = federated_language.FunctionType(
        parameter_type, result_type
    )
    tensorflow_proto = pb.TensorFlow(
        graph_def=serialization_utils.pack_graph_def(graph.as_graph_def()),
        parameter=None,
        session_token_tensor_name=session_token_placeholder.name,
        result=result_binding,
    )
    serialized_type = type_signature.to_proto()
    computation_proto = pb.Computation(
        type=serialized_type, tensorflow=tensorflow_proto
    )
    init_op, result = tensorflow_utils.deserialize_and_call_tf_computation(
        computation_proto,
        None,
        tf.compat.v1.get_default_graph(),
        '',
        tf.constant('bogus_token'),
    )
    self.assertTrue(tf.is_tensor(result))
    with tf.compat.v1.Session() as sess:
      if init_op:
        sess.run(init_op)
      result_val = sess.run(result)
    self.assertEqual(result_val, b'bogus_token')


if __name__ == '__main__':
  absltest.main()
