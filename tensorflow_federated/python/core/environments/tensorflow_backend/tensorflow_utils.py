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
"""Utilities for interacting with and manipulating TensorFlow graphs."""

import collections
from collections.abc import Iterable, Mapping
import itertools
import typing
from typing import Optional, Union

import attrs
import federated_language
from federated_language.proto import computation_pb2 as pb
import numpy as np
import tensorflow as tf

from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.environments.tensorflow_backend import graph_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import serialization_utils
from tensorflow_federated.python.core.environments.tensorflow_backend import type_conversions
from tensorflow_federated.python.core.environments.tensorflow_frontend import tensorflow_types

_TENSOR_REPRESENTATION_TYPES = (
    # Python native types
    str,
    int,
    float,
    bool,
    bytes,
    # Numpy data types
    np.generic,
    np.ndarray,
)


def stamp_parameter_in_graph(parameter_name, parameter_type, graph):
  """Stamps a parameter of a given type in the given tf.Graph instance.

  Tensors are stamped as placeholders, sequences are stamped as data sets
  constructed from string tensor handles, and named tuples are stamped by
  independently stamping their elements.

  Args:
    parameter_name: The suggested (string) name of the parameter to use in
      determining the names of the graph components to construct. The names that
      will actually appear in the graph are not guaranteed to be based on this
      suggested name, and may vary, e.g., due to existing naming conflicts, but
      a best-effort attempt will be made to make them similar for ease of
      debugging.
    parameter_type: The type of the parameter to stamp. Must be either an
      instance of federated_language.Type (or convertible to it), or None.
    graph: The instance of tf.Graph to stamp in.

  Returns:
    A tuple (val, binding), where 'val' is a Python object (such as a dataset,
    a placeholder, or a `structure.Struct` that represents a named
    tuple) that represents the stamped parameter for use in the body of a Python
    function that consumes this parameter, and the 'binding' is an instance of
    TensorFlow.Binding that indicates how parts of the type signature relate
    to the tensors and ops stamped into the graph.

  Raises:
    TypeError: If the arguments are of the wrong federated_language.
    ValueError: If the parameter type cannot be stamped in a TensorFlow graph.
  """
  py_typecheck.check_type(parameter_name, str)
  py_typecheck.check_type(graph, tf.Graph)
  if parameter_type is None:
    return (None, None)
  parameter_type = tensorflow_types.to_type(parameter_type)
  if isinstance(parameter_type, federated_language.TensorType):
    with graph.as_default():
      placeholder = tf.compat.v1.placeholder(
          dtype=parameter_type.dtype,
          shape=parameter_type.shape,
          name=parameter_name,
      )
      binding = pb.TensorFlow.Binding(
          tensor=pb.TensorFlow.TensorBinding(tensor_name=placeholder.name)
      )
      return (placeholder, binding)
  elif isinstance(parameter_type, federated_language.StructType):
    # The parameter_type could be a StructTypeWithPyContainer, however, we
    # ignore that for now. Instead, the proper containers will be inserted at
    # call time by
    # `federated_language.framework.wrap_as_zero_or_one_arg_callable`.
    if not parameter_type:
      # Stamps whimsy element to "populate" graph, as TensorFlow does not
      # support empty graphs.
      whimsy_tensor = tf.no_op()
      del whimsy_tensor  # Unused
    element_name_value_pairs = []
    element_bindings = []
    for name, element_type in parameter_type.items():
      e_val, e_binding = stamp_parameter_in_graph(
          '{}_{}'.format(parameter_name, name), element_type, graph
      )
      element_name_value_pairs.append((name, e_val))
      element_bindings.append(e_binding)
    return (
        structure.Struct(element_name_value_pairs),
        pb.TensorFlow.Binding(
            struct=pb.TensorFlow.StructBinding(element=element_bindings)
        ),
    )
  elif isinstance(parameter_type, federated_language.SequenceType):
    with graph.as_default():
      with tf.device('/device:cpu:0'):
        variant_tensor = tf.compat.v1.placeholder(tf.variant, shape=[])
        ds = _make_dataset_from_variant_tensor(
            variant_tensor, parameter_type.element
        )
    return (
        ds,
        pb.TensorFlow.Binding(
            sequence=pb.TensorFlow.SequenceBinding(
                variant_tensor_name=variant_tensor.name
            )
        ),
    )
  else:
    raise ValueError(
        'Parameter type component {!r} cannot be stamped into a TensorFlow '
        'graph.'.format(parameter_type)
    )


def _make_dataset_from_variant_tensor(variant_tensor, type_spec):
  """Constructs a `tf.data.Dataset` from a variant tensor and type spec.

  Args:
    variant_tensor: The variant tensor that represents the dataset.
    type_spec: The type spec of elements of the data set, either an instance of
      `types.Type` or something convertible to it.

  Returns:
    A corresponding instance of `tf.data.Dataset`.

  Raises:
    TypeError: If the arguments are of the wrong types.
  """
  if not tf.is_tensor(variant_tensor):
    raise TypeError(
        'Expected `variant_tensor` to be a tensor, found {}.'.format(
            py_typecheck.type_string(type(variant_tensor))
        )
    )
  if variant_tensor.dtype != tf.variant:
    raise TypeError(
        'Expected `variant_tensor` to be of a variant type, found {}.'.format(
            variant_tensor.dtype
        )
    )
  with tf.device('/device:cpu:0'):
    return tf.data.experimental.from_variant(
        variant_tensor,
        structure=(
            type_conversions.type_to_tf_structure(
                tensorflow_types.to_type(type_spec)
            )
        ),
    )


class InvalidGraphResultError(TypeError):
  """Error creating a TF type and binding for a TF graph result."""


class DictionaryKeyMustBeStringError(InvalidGraphResultError):
  """Error due to a non-string dictionary key in a TF graph result structure."""


class InvalidDatasetElementSpecError(InvalidGraphResultError):
  """Error due to invalid `element_spec` in dataset result of TF graph."""


class UnsupportedGraphResultError(InvalidGraphResultError):
  """Error due to unsupported type in a TF graph result structure."""


def capture_result_from_graph(
    result: object,
    graph: tf.Graph,
) -> tuple[federated_language.Type, pb.TensorFlow.Binding]:
  """Captures a result stamped into a tf.Graph as a type signature and binding.

  Args:
    result: The result to capture, a Python object that is composed of tensors,
      possibly nested within Python structures such as dictionaries, lists,
      tuples, or named tuples.
    graph: The instance of tf.Graph to use.

  Returns:
    A tuple (type_spec, binding), where 'type_spec' is an instance of
    federated_language.Type that describes the type of the result, and 'binding'
    is an instance of TensorFlow.Binding that indicates how parts of the result
    type relate to the tensors and ops that appear in the result.

  Raises:
    DictionaryKeyMustBeStringError: If `result` contains a dictionary with a
      non-`str` key.
    InvalidDatasetElementSpecError: If `result` contains a dataset with an
      `element_spec` that is unsupported.
    UnsupportedGraphResultError: If `result` contains a value which for which
      conversion in to a `tf.Graph` output is not supported.
  """

  def _get_bindings_for_elements(
      name_value_pairs: Iterable[tuple[str, object]],
      graph: tf.Graph,
      container_type: Optional[type[object]],
  ) -> tuple[federated_language.Type, pb.TensorFlow.Binding]:
    """Build `(type_spec, binding)` tuple for name value pairs."""
    element_name_type_binding_triples = [
        ((k,) + capture_result_from_graph(v, graph))
        for k, v in name_value_pairs
    ]
    type_members = []
    for e in element_name_type_binding_triples:
      type_member = (e[0], e[1]) if e[0] else e[1]
      type_members.append(type_member)
    if container_type:
      type_spec = federated_language.StructWithPythonType(
          type_members, container_type=container_type
      )
    else:
      type_spec = federated_language.StructType(type_members)
    binding = pb.TensorFlow.Binding(
        struct=pb.TensorFlow.StructBinding(
            element=[e[2] for e in element_name_type_binding_triples]
        )
    )
    return type_spec, binding

  # TODO: b/113112885 - The emerging extensions for serializing SavedModels may
  # end up introducing similar concepts of bindings, etc., we should look here
  # into the possibility of reusing some of that code when it's available.
  if isinstance(result, _TENSOR_REPRESENTATION_TYPES):
    with graph.as_default():
      result = tf.constant(result)
  # TODO: b/224484886 - Downcasting to all handled types.
  result = typing.cast(
      Union[
          py_typecheck.SupportsNamedTuple,
          structure.Struct,
          Mapping[str, object],
          list[object],
          tuple[object, ...],
          tf.data.Dataset,
      ],
      result,
  )
  if tf.is_tensor(result):
    if hasattr(result, 'read_value'):
      # We have a tf.Variable-like result, get a proper tensor to fetch.
      with graph.as_default():
        result = result.read_value()
    else:
      # Otherwise we insert an identity. TensorFlow does not allow the same
      # tensor to appear in both feeds and fetches, which can occur if the
      # federated_language.Computation is only performing a selection from a
      # structure.
      with graph.as_default():
        result = tf.identity(result)
    # `tf.is_tensor` returns true for some things that are not actually single
    # `tf.Tensor`s, including `tf.sparse.SparseTensor`s and `tf.RaggedTensor`s.
    if isinstance(result, tf.RaggedTensor):
      name_value_pairs = (
          ('flat_values', result.flat_values),
          ('nested_row_splits', result.nested_row_splits),
      )
      return _get_bindings_for_elements(
          name_value_pairs, graph, tf.RaggedTensor
      )
    elif isinstance(result, tf.sparse.SparseTensor):
      name_value_pairs = (
          ('indices', result.indices),
          ('values', result.values),
          ('dense_shape', result.dense_shape),
      )
      return _get_bindings_for_elements(
          name_value_pairs, graph, tf.sparse.SparseTensor
      )
    else:
      if result.dtype.base_dtype == tf.string:
        dtype = np.str_
      else:
        dtype = result.dtype.base_dtype.as_numpy_dtype
      if result.shape.rank is not None:
        shape = result.shape.as_list()
      else:
        shape = None
      return (
          federated_language.TensorType(dtype, shape),
          pb.TensorFlow.Binding(
              tensor=pb.TensorFlow.TensorBinding(tensor_name=result.name)
          ),
      )
  elif isinstance(result, py_typecheck.SupportsNamedTuple):
    # Special handling needed for collections.namedtuples since they do not have
    # anything in the way of a shared base class. Note we don't want to rely on
    # the fact that collections.namedtuples inherit from 'tuple' because we'd be
    # failing to retain the information about naming of tuple members.
    name_value_pairs = result._asdict().items()
    return _get_bindings_for_elements(name_value_pairs, graph, type(result))
  elif attrs.has(type(result)):
    name_value_pairs = attrs.asdict(result, recurse=False).items()
    return _get_bindings_for_elements(name_value_pairs, graph, type(result))
  elif isinstance(result, structure.Struct):
    return _get_bindings_for_elements(
        structure.to_elements(result), graph, None
    )
  elif isinstance(result, Mapping):
    for key in result:
      if not isinstance(key, str):
        key_type_str = py_typecheck.type_string(type(key))
        raise DictionaryKeyMustBeStringError(
            'Dictionaries returned from TensorFlow graphs must be of type '
            f'`str`, but found key `{key}` of type `{key_type_str}`.'
        )
    name_value_pairs = result.items()
    return _get_bindings_for_elements(name_value_pairs, graph, type(result))
  elif isinstance(result, (list, tuple)):
    element_type_binding_pairs = [
        capture_result_from_graph(e, graph) for e in result
    ]
    return (
        federated_language.StructWithPythonType(
            [e[0] for e in element_type_binding_pairs], type(result)
        ),
        pb.TensorFlow.Binding(
            struct=pb.TensorFlow.StructBinding(
                element=[e[1] for e in element_type_binding_pairs]
            )
        ),
    )
  elif isinstance(result, tf.data.Dataset):
    try:
      element_type = tensorflow_types.to_type(result.element_spec)
    except TypeError as e:
      raise InvalidDatasetElementSpecError(
          'Dataset has `element_spec` which is not a valid TFF type.\n'
          f'Found `element_spec`: {result.element_spec}\n'
          f'which is not a valid TFF type: {str(e)}'
      ) from None

    # This variant tensor needs an identity added to ensure that parameter and
    # result bindings in our graphdefs are distinct. A similar operation is
    # performed in generation of tf.function.
    # We additionally pin this return dataset to CPU to prevent placer from
    # attempting to copy the identity operation with dataset input (CPU only) to
    # GPU.
    with graph.as_default():
      with tf.device('/device:cpu:0'):
        variant_tensor = tf.identity(tf.data.experimental.to_variant(result))
    return (
        federated_language.SequenceType(element_type),
        pb.TensorFlow.Binding(
            sequence=pb.TensorFlow.SequenceBinding(
                variant_tensor_name=variant_tensor.name
            )
        ),
    )
  else:
    result_type_str = py_typecheck.type_string(type(result))
    raise UnsupportedGraphResultError(
        f'Cannot capture a result of an unsupported type {result_type_str}.'
    )


def _compute_map_from_bindings(source, target):
  """Computes a dictionary for renaming tensors from a matching bindings pair.

  Args:
    source: An instance of `pb.TensorFlow.Binding` that contains names of
      tensors that will form the keys in the dictionary.
    target: An instance of `pb.TensorFlow.Binding` that contains names of
      tensors that will form the values in the dictionary. The structure of this
      binding must be identical as that of the `source`.

  Returns:
    A dictionary mapping names of tensors in `source` to names of the
    tensors in the corresponding parts of `target`.

  Raises:
    TypeError: If the arguments are of the wrong federated_language.
    ValueError: If the bindings have mismatching structures.
  """
  py_typecheck.check_type(source, pb.TensorFlow.Binding)
  py_typecheck.check_type(target, pb.TensorFlow.Binding)
  source_oneof = source.WhichOneof('binding')
  target_oneof = target.WhichOneof('binding')
  if source_oneof != target_oneof:
    raise ValueError(
        'Source and target binding variants mismatch: {} vs. {}'.format(
            source_oneof, target_oneof
        )
    )
  if source_oneof == 'tensor':
    return collections.OrderedDict(
        [(str(source.tensor.tensor_name), str(target.tensor.tensor_name))]
    )
  elif source_oneof == 'sequence':
    sequence_oneof = source.sequence.WhichOneof('binding')
    if target.sequence.WhichOneof('binding') != sequence_oneof:
      raise ValueError(
          'Source and target sequence bindings mismatch: {} vs. {}'.format(
              sequence_oneof, target.sequence.WhichOneof('binding')
          )
      )
    if sequence_oneof == 'variant_tensor_name':
      return collections.OrderedDict([
          (
              str(source.sequence.variant_tensor_name),
              str(target.sequence.variant_tensor_name),
          ),
      ])
    else:
      raise ValueError('Unsupported sequence binding {}'.format(sequence_oneof))
  elif source_oneof == 'struct':
    if len(source.struct.element) != len(target.struct.element):
      raise ValueError(
          'Source and target binding tuple lengths mismatch: {} vs. {}.'.format(
              len(source.struct.element), len(target.struct.element)
          )
      )
    else:
      result = collections.OrderedDict()
      for source_element, target_element in zip(
          source.struct.element, target.struct.element
      ):
        result.update(
            _compute_map_from_bindings(source_element, target_element)
        )
      return result
  else:
    raise ValueError("Unsupported type of binding '{}'.".format(source_oneof))


def extract_tensor_names_from_binding(binding):
  """Returns a list of tensor names extracted from a given binding.

  Args:
    binding: An instance of `pb.TensorFlow.Binding`.

  Returns:
    All tensor names that appear in `binding`.
  """
  py_typecheck.check_type(binding, pb.TensorFlow.Binding)
  binding_oneof = binding.WhichOneof('binding')
  if binding_oneof == 'tensor':
    return [str(binding.tensor.tensor_name)]
  elif binding_oneof == 'sequence':
    sequence_oneof = binding.sequence.WhichOneof('binding')
    if sequence_oneof == 'variant_tensor_name':
      return [str(binding.sequence.variant_tensor_name)]
    else:
      raise ValueError('Unsupported sequence binding {}'.format(sequence_oneof))
  elif binding_oneof == 'struct':
    return list(
        itertools.chain.from_iterable([
            extract_tensor_names_from_binding(e) for e in binding.struct.element
        ])
    )
  else:
    raise ValueError("Unsupported type of binding '{}'.".format(binding_oneof))


def _assemble_result_from_graph(type_spec, binding, output_map):
  """Assembles a result stamped into a `tf.Graph` given type signature/binding.

  This method does roughly the opposite of `capture_result_from_graph`, in that
  whereas `capture_result_from_graph` starts with a single structured object
  made up of tensors and computes its type and bindings, this method starts
  with the type/bindings and constructs a structured object made up of tensors.

  Args:
    type_spec: The type signature of the result to assemble, an instance of
      `types.Type` or something convertible to it.
    binding: The binding that relates the type signature to names of tensors in
      the graph, an instance of `pb.TensorFlow.Binding`.
    output_map: The mapping from tensor names that appear in the binding to
      actual stamped tensors (possibly renamed during import).

  Returns:
    The assembled result, a Python object that is composed of tensors, possibly
    nested within Python structures such as anonymous tuples.

  Raises:
    TypeError: If the argument or any of its parts are of an uexpected type.
    ValueError: If the arguments are invalid or inconsistent witch other, e.g.,
      the type and binding don't match, or the tensor is not found in the map.
  """
  type_spec = tensorflow_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, federated_language.Type)
  py_typecheck.check_type(binding, pb.TensorFlow.Binding)
  py_typecheck.check_type(output_map, dict)
  for k, v in output_map.items():
    py_typecheck.check_type(k, str)
    if not tf.is_tensor(v):
      raise TypeError(
          'Element with key {} in the output map is {}, not a tensor.'.format(
              k, py_typecheck.type_string(type(v))
          )
      )

  binding_oneof = binding.WhichOneof('binding')
  if isinstance(type_spec, federated_language.TensorType):
    if binding_oneof != 'tensor':
      raise ValueError(
          'Expected a tensor binding, found {}.'.format(binding_oneof)
      )
    elif binding.tensor.tensor_name not in output_map:
      raise ValueError(
          'Tensor named {} not found in the output map.'.format(
              binding.tensor.tensor_name
          )
      )
    else:
      tensor_name = binding.tensor.tensor_name
      tensor = output_map[tensor_name]
      inferred_type = type_conversions.tensorflow_infer_type(tensor)
      if not type_spec.is_assignable_from(inferred_type):
        raise ValueError(
            f'Type mismatch loading graph result tensor {tensor} '
            f'(named "{tensor_name}").\n'
            'This may have been caused by a use of `tf.set_shape`.\n'
            'Prefer usage of `tf.ensure_shape` to `tf.set_shape`.'
        )
      return tensor
  elif isinstance(type_spec, federated_language.StructType):
    if binding_oneof != 'struct':
      raise ValueError(
          'Expected a struct binding, found {}.'.format(binding_oneof)
      )
    else:
      type_elements = list(type_spec.items())
      if len(binding.struct.element) != len(type_elements):
        raise ValueError(
            'Mismatching tuple sizes in type ({}) and binding ({}).'.format(
                len(type_elements), len(binding.struct.element)
            )
        )
      result_elements = []
      for (element_name, element_type), element_binding in zip(
          type_elements, binding.struct.element
      ):
        element_object = _assemble_result_from_graph(
            element_type, element_binding, output_map
        )
        result_elements.append((element_name, element_object))
      if type_spec.python_container is None:
        return structure.Struct(result_elements)
      container_type = type_spec.python_container
      if isinstance(
          container_type, py_typecheck.SupportsNamedTuple
      ) or attrs.has(container_type):
        return container_type(**dict(result_elements))
      return container_type(result_elements)  # pylint: disable=too-many-function-args
  elif isinstance(type_spec, federated_language.SequenceType):
    if binding_oneof != 'sequence':
      raise ValueError(
          'Expected a sequence binding, found {}.'.format(binding_oneof)
      )
    else:
      sequence_oneof = binding.sequence.WhichOneof('binding')
      if sequence_oneof == 'variant_tensor_name':
        variant_tensor = output_map[binding.sequence.variant_tensor_name]
        return _make_dataset_from_variant_tensor(
            variant_tensor, type_spec.element
        )
      else:
        raise ValueError(
            "Unsupported sequence binding '{}'.".format(sequence_oneof)
        )
  else:
    raise ValueError("Unsupported type '{}'.".format(type_spec))


def _make_empty_list_structure_for_element_type_spec(type_spec):
  """Creates a nested structure of empty Python lists for `type_spec`.

  This function prepares a nested structure made of `collections.OrderedDict`s
  and Python `tuple`s at the intermediate (non-leaf) levels, and that has empty
  Python `list`s at the leaf level, with the shape of the structure matching
  that of `type_spec`. This structure is used to accumulate elemnts of a data
  set for ingestion by `tf.data.Dataset.from_tensor_slices`.

  Args:
    type_spec: An instance of `federated_language.Type` or something convertible
      to it that consists of only tensor and named tuple types, and in which
      rach of the named tuples either have all or none of their elements named.

  Returns:
    The nested structure, as described above.

  Raises:
    TypeError: If the `type_spec` is not of a form described above.
  """
  type_spec = tensorflow_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, federated_language.Type)
  if isinstance(type_spec, federated_language.TensorType):
    return []
  elif isinstance(type_spec, federated_language.StructType):
    elements = list(type_spec.items())
    if all(k is not None for k, _ in elements):
      return collections.OrderedDict([
          (k, _make_empty_list_structure_for_element_type_spec(v))
          for k, v in elements
      ])
    elif all(k is None for k, _ in elements):
      return tuple([
          _make_empty_list_structure_for_element_type_spec(v)
          for _, v in elements
      ])
    else:
      raise TypeError(
          'Expected a named tuple type with either all elements named or all '
          'unnamed, got {}.'.format(type_spec)
      )
  else:
    raise TypeError(
        'Expected a tensor or named tuple type, found {}.'.format(type_spec)
    )


def _make_whimsy_element_for_type_spec(type_spec, none_dim_replacement=0):
  """Creates ndarray of zeros corresponding to `type_spec`.

  Returns a list containing this ndarray, whose type is *compatible* with, not
  necessarily equal to, `type_spec`. This is due to the fact that some
  dimensions of `type_spec` may be indeterminate, representing compatibility
  of `type_spec` with any number (e.g. leaving a batch dimension indeterminate
  to signify compatibility with batches of any size). However a concrete
  structure (like the ndarray) must have specified sizes for its dimensions.
  So we construct a whimsy element where any `None` dimensions of the shape
  of `type_spec` are replaced with the value `none_dim_replacement`.The
  default value of 0 therefore returns a whimsy element of minimal size which
  matches `type_spec`.

  Args:
    type_spec: A `federated_language.Type`, or something convertible to it.
    none_dim_replacement: `int` with which to replace any unspecified tensor
      dimensions.

  Returns:
    Returns possibly nested `numpy ndarray`s containing all zeros: a single
    `ndarray` if `type_spec` is a `federated_language.TensorType` and a list
    of such arrays if  `type_spec` is `federated_language.StructType`.
    This data structure is of the minimal size necessary in order to be
    compatible with `type_spec`.
  """
  type_spec = tensorflow_types.to_type(type_spec)

  def _predicate(type_spec: federated_language.Type) -> bool:
    return isinstance(
        type_spec,
        (
            federated_language.TensorType,
            federated_language.StructType,
        ),
    )

  if not federated_language.framework.type_contains_only(type_spec, _predicate):
    raise ValueError(
        'Cannot construct array for TFF type containing anything '
        'other than `federated_language.TensorType` or '
        '`federated_language.StructType`; you have passed the '
        'type {}'.format(type_spec)
    )
  py_typecheck.check_type(none_dim_replacement, int)
  if none_dim_replacement < 0:
    raise ValueError(
        'Please pass nonnegative integer argument as `none_dim_replacement`.'
    )

  def _handle_none_dimension(x):
    if x is None or (isinstance(x, tf.compat.v1.Dimension) and x.value is None):
      return none_dim_replacement
    return x

  if isinstance(type_spec, federated_language.TensorType):
    whimsy_shape = [_handle_none_dimension(x) for x in type_spec.shape]
    if type_spec.dtype == np.str_:
      return np.empty(whimsy_shape, dtype=np.str_)
    return np.zeros(whimsy_shape, type_spec.dtype)
  elif isinstance(type_spec, federated_language.StructType):
    elements = list(type_spec.items())
    elem_list = []
    for _, elem_type in elements:
      elem_list.append(_make_whimsy_element_for_type_spec(elem_type))
    return elem_list


def _append_to_list_structure_for_element_type_spec(nested, value, type_spec):
  """Adds an element `value` to `nested` lists for `type_spec`.

  This function appends tensor-level constituents of an element `value` to the
  lists created by `_make_empty_list_structure_for_element_type_spec`. The
  nested structure of `value` must match that created by the above function,
  and consistent with `type_spec`.

  Args:
    nested: Output of `_make_empty_list_structure_for_element_type_spec`.
    value: A value (Python object) that a hierarchical structure of dictionary,
      list, and other containers holding tensor-like items that matches the
      hierarchy of `type_spec`.
    type_spec: An instance of `federated_language.Type` or something convertible
      to it, as in `_make_empty_list_structure_for_element_type_spec`.

  Raises:
    TypeError: If the `type_spec` is not of a form described above, or the value
      is not of a type compatible with `type_spec`.
  """
  if value is None:
    return
  type_spec = tensorflow_types.to_type(type_spec)
  # TODO: b/113116813 - This could be made more efficient, but for now we won't
  # need to worry about it as this is an odd corner case.
  if isinstance(value, structure.Struct):
    elements = structure.to_elements(value)
    if all(k is not None for k, _ in elements):
      value = collections.OrderedDict(elements)
    elif all(k is None for k, _ in elements):
      value = tuple([v for _, v in elements])
    else:
      raise TypeError(
          'Expected an anonymous tuple to either have all elements named or '
          'all unnamed, got {}.'.format(value)
      )
  if isinstance(type_spec, federated_language.TensorType):
    if not isinstance(nested, list):
      raise TypeError(f'Expected `nested` to be a `list`, found {type(nested)}')
    # Convert the members to tensors to ensure that they are properly
    # typed and grouped before being passed to
    # tf.data.Dataset.from_tensor_slices.
    nested.append(tf.convert_to_tensor(value, type_spec.dtype))
  elif isinstance(type_spec, federated_language.StructType):
    elements = list(type_spec.items())
    if isinstance(nested, collections.OrderedDict):
      if isinstance(value, py_typecheck.SupportsNamedTuple):
        # In Python 3.8 and later `_asdict` no longer return OrdereDict, rather
        # a regular `dict`.
        value = collections.OrderedDict(value._asdict())
      if isinstance(value, dict):
        if set(value.keys()) != set(k for k, _ in elements):
          raise TypeError(
              'Value {} does not match type {}.'.format(value, type_spec)
          )
        for elem_name, elem_type in elements:
          _append_to_list_structure_for_element_type_spec(
              nested[elem_name], value[elem_name], elem_type
          )
      elif isinstance(value, (list, tuple)):
        if len(value) != len(elements):
          raise TypeError(
              'Value {} does not match type {}.'.format(value, type_spec)
          )
        for idx, (elem_name, elem_type) in enumerate(elements):
          _append_to_list_structure_for_element_type_spec(
              nested[elem_name], value[idx], elem_type
          )
      else:
        raise TypeError(
            'Unexpected type of value {} for TFF type {}.'.format(
                py_typecheck.type_string(type(value)), type_spec
            )
        )
    elif isinstance(nested, tuple):
      py_typecheck.check_type(value, (list, tuple))
      if len(value) != len(elements):
        raise TypeError(
            'Value {} does not match type {}.'.format(value, type_spec)
        )
      for idx, (_, elem_type) in enumerate(elements):
        _append_to_list_structure_for_element_type_spec(
            nested[idx], value[idx], elem_type
        )
    else:
      raise TypeError(
          'Invalid nested structure, unexpected container type {}.'.format(
              py_typecheck.type_string(type(nested))
          )
      )
  else:
    raise TypeError(
        'Expected a tensor or named tuple type, found {}.'.format(type_spec)
    )


def _replace_empty_leaf_lists_with_numpy_arrays(lists, type_spec):
  """Replaces empty leaf lists in `lists` with numpy arrays.

  This function is primarily used to ensure that an appropriate TF dtype is
  inferrable for a structure, even if no elements are actually present.

  Args:
    lists: Output of `_make_empty_list_structure_for_element_type_spec`.
    type_spec: An instance of `federated_language.Type` or something convertible
      to it, as in `_make_empty_list_structure_for_element_type_spec`.

  Returns:
    The transformed version of `structure`.

  Raises:
    TypeError: If the `type_spec` is not of a form described above, or if
      `lists` is not of a type compatible with `type_spec`.
  """
  type_spec = tensorflow_types.to_type(type_spec)
  py_typecheck.check_type(type_spec, federated_language.Type)
  if isinstance(type_spec, federated_language.TensorType):
    py_typecheck.check_type(lists, list)
    if lists:
      return lists
    else:
      return np.array([], dtype=type_spec.dtype)
  elif isinstance(type_spec, federated_language.StructType):
    elements = list(type_spec.items())
    if isinstance(lists, collections.OrderedDict):
      to_return = []
      for elem_name, elem_type in elements:
        elem_val = _replace_empty_leaf_lists_with_numpy_arrays(
            lists[elem_name], elem_type
        )
        to_return.append((elem_name, elem_val))
      return collections.OrderedDict(to_return)
    elif isinstance(lists, tuple):
      to_return = []
      for idx, (_, elem_type) in enumerate(elements):
        elem_val = _replace_empty_leaf_lists_with_numpy_arrays(
            lists[idx], elem_type
        )
        to_return.append(elem_val)
      return tuple(to_return)
    else:
      raise TypeError(
          'Invalid nested structure, unexpected container type {}.'.format(
              py_typecheck.type_string(type(lists))
          )
      )
  else:
    raise TypeError(
        'Expected a tensor or struct type, found {}.'.format(type_spec)
    )


def make_data_set_from_elements(graph, elements, element_type):
  """Creates a `tf.data.Dataset` in `graph` from explicitly listed `elements`.

  Note: The underlying implementation attempts to use the
  `tf.data.Dataset.from_tensor_slices() method to build the data set quickly,
  but this doesn't always work. The typical scenario where it breaks is one
  with data set being composed of unequal batches. Typically, only the last
  batch is odd, so on the first attempt, we try to construct two data sets,
  one from all elements but the last one, and one from the last element, then
  concatenate the two. In the unlikely case that this fails (e.g., because
  all data set elements are batches of unequal sizes), we revert to the slow,
  but reliable method of constructing data sets from singleton elements, and
  then concatenating them all.

  Args:
    graph: The graph in which to construct the `tf.data.Dataset`, or `None` if
      the construction is to happen in the eager context.
    elements: A list of elements.
    element_type: The type of elements.

  Returns:
    The constructed `tf.data.Dataset` instance.

  Raises:
    TypeError: If element types do not match `element_type`.
    ValueError: If the elements are of incompatible types and shapes, or if
      no graph was specified outside of the eager context.
  """
  # Note: We allow the graph to be `None` to allow this function to be used in
  # the eager context.
  if graph is not None:
    py_typecheck.check_type(graph, tf.Graph)
  elif not tf.executing_eagerly():
    raise ValueError('Only in eager context may the graph be `None`.')
  py_typecheck.check_type(elements, list)
  element_type = tensorflow_types.to_type(element_type)
  py_typecheck.check_type(element_type, federated_language.Type)

  def _make(element_subset):
    lists = _make_empty_list_structure_for_element_type_spec(element_type)
    for el in element_subset:
      _append_to_list_structure_for_element_type_spec(lists, el, element_type)
    tensor_slices = _replace_empty_leaf_lists_with_numpy_arrays(
        lists, element_type
    )
    return tf.data.Dataset.from_tensor_slices(tensor_slices)

  def _work():
    if not elements:
      # Just return an empty data set with the appropriate types.
      whimsy_element = _make_whimsy_element_for_type_spec(element_type)
      ds = _make([whimsy_element]).take(0)
    elif len(elements) == 1:
      ds = _make(elements)
    else:
      try:
        # It is common for the last element to be a batch of a size different
        # from all the preceding batches. With this in mind, we proactively
        # single out the last element (optimizing for the common case).
        ds = _make(elements[0:-1]).concatenate(_make(elements[-1:]))
      except ValueError:
        # In case elements beyond just the last one are of unequal shapes, we
        # may have failed (the most likely cause), so fall back onto the slow
        # process of constructing and joining data sets from singletons. Not
        # optimizing this for now, as it's very unlikely in scenarios
        # we're targeting.
        #
        # Note: this will not remain `None` because `element`s is not empty.
        ds = None
        ds = typing.cast(tf.data.Dataset, ds)
        for i in range(len(elements)):
          singleton_ds = _make(elements[i : i + 1])
          ds = singleton_ds if ds is None else ds.concatenate(singleton_ds)
    ds_element_type = tensorflow_types.to_type(ds.element_spec)
    if not element_type.is_assignable_from(ds_element_type):  # pytype: disable=attribute-error
      raise TypeError(
          'Failure during data set construction, expected elements of type {}, '
          'but the constructed data set has elements of type {}.'.format(
              element_type, ds_element_type
          )
      )
    return ds

  if graph is not None:
    with graph.as_default():
      return _work()
  else:
    return _work()


def fetch_value_in_session(sess, value):
  """Fetches `value` in `session`.

  Args:
    sess: The session in which to perform the fetch (as a single run).
    value: A Python object of a form analogous to that constructed by the
      function `_assemble_result_from_graph`, made of tensors and anononymous
      tuples, or a `tf.data.Dataset`.

  Returns:
    A Python object with structure similar to `value`, but with tensors
    replaced with their values, and data sets replaced with lists of their
    elements, all fetched with a single call `session.run()`.

  Raises:
    ValueError: If `value` is not a `tf.data.Dataset` or not a structure of
      tensors and anonoymous tuples.
  """
  py_typecheck.check_type(sess, tf.compat.v1.Session)
  # TODO: b/113123634 - Investigate handling `list`s and `tuple`s of
  # `tf.data.Dataset`s and what the API would look like to support this.
  if isinstance(value, tf.data.Dataset):
    with sess.graph.as_default():
      iterator = tf.compat.v1.data.make_one_shot_iterator(value)
      next_element = iterator.get_next()
    elements = []
    while True:
      try:
        elements.append(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        break
    return elements
  else:
    flattened_value = structure.flatten(value)
    dataset_results = {}
    flat_tensors = []
    for idx, v in enumerate(flattened_value):
      if isinstance(v, tf.data.Dataset):
        dataset_tensors = fetch_value_in_session(sess, v)
        if not dataset_tensors:
          # An empty list has been returned; we must pack the shape information
          # back in or the result won't typecheck.
          element_structure = v.element_spec
          whimsy_elem = _make_whimsy_element_for_type_spec(element_structure)
          dataset_tensors = [whimsy_elem]
        dataset_results[idx] = dataset_tensors
      elif tf.is_tensor(v):
        flat_tensors.append(v)
      else:
        raise ValueError('Unsupported value type {}.'.format(v))
    # Note that `flat_tensors` could be an empty tuple, but it could also be a
    # list of empty tuples.
    if flat_tensors or any(x for x in flat_tensors):
      flat_computed_tensors = sess.run(flat_tensors)
    else:
      flat_computed_tensors = flat_tensors
    flattened_results = _interleave_dataset_results_and_tensors(
        dataset_results, flat_computed_tensors
    )

    def _to_unicode(v):
      if isinstance(v, bytes):
        return v.decode('utf-8')
      return v

    if tf.is_tensor(value) and value.dtype == tf.string:
      flattened_results = [_to_unicode(result) for result in flattened_results]
    return structure.pack_sequence_as(value, flattened_results)


def _interleave_dataset_results_and_tensors(dataset_results, flat_run_tensors):
  flattened_results = []
  for idx in range(len(dataset_results) + len(flat_run_tensors)):
    if dataset_results.get(idx):
      flattened_results.append(dataset_results[idx])
    else:
      flattened_results.append(flat_run_tensors.pop(0))
  return flattened_results


def coerce_dataset_elements_to_tff_type_spec(
    dataset: tf.data.Dataset, element_type: federated_language.Type
) -> tf.data.Dataset:
  """Map the elements of a dataset to a specified type.

  This is used to coerce a `tf.data.Dataset` that may have lost the ordering
  of dictionary keys back into a `collections.OrderedDict` (required by TFF).

  Args:
    dataset: a `tf.data.Dataset` instance.
    element_type: a `federated_language.Type` specifying the type of the
      elements of `dataset`. Must be a `federated_language.TensorType` or
      `federated_language.StructType`.

  Returns:
    A `tf.data.Dataset` whose output types are compatible with
    `element_type`.

  Raises:
    ValueError: if the elements of `dataset` cannot be coerced into
      `element_type`.
  """
  py_typecheck.check_type(element_type, federated_language.Type)
  if isinstance(element_type, federated_language.TensorType):
    return dataset
  elif isinstance(element_type, federated_language.StructWithPythonType):
    py_type = element_type.python_container
    if py_type is tf.RaggedTensor or py_type is tf.sparse.SparseTensor:
      return dataset

  # This is a similar to `reference_context.to_representation_for_type`,
  # look for opportunities to consolidate?
  def _to_representative_value(type_spec, elements):
    """Convert to a container to a type understood by TF and TFF."""
    if isinstance(type_spec, federated_language.TensorType):
      return elements
    elif isinstance(type_spec, federated_language.StructWithPythonType):
      if tf.is_tensor(elements):
        # In this case we have a singleton tuple tensor that may have been
        # unwrapped by tf.data.
        elements = [elements]
      py_type = type_spec.python_container
      if py_type is tf.RaggedTensor or py_type is tf.sparse.SparseTensor:
        return elements

      field_types = type_spec.items()
      if issubclass(py_type, Mapping) or attrs.has(py_type):
        values = collections.OrderedDict(
            (name, _to_representative_value(field_type, elements[name]))
            for name, field_type in field_types
        )
        return py_type(**values)
      else:
        values = [
            _to_representative_value(field_type, e)
            for (_, field_type), e in zip(field_types, elements)
        ]
        if isinstance(py_type, py_typecheck.SupportsNamedTuple):
          return py_type(*values)
        return py_type(values)  # pylint: disable=too-many-function-args
    elif isinstance(type_spec, federated_language.StructType):
      field_types = list(type_spec.items())
      is_all_named = all([name is not None for name, _ in field_types])
      if is_all_named:
        if isinstance(elements, py_typecheck.SupportsNamedTuple):
          values = collections.OrderedDict(
              (name, _to_representative_value(field_type, e))
              for (name, field_type), e in zip(field_types, elements)
          )
          return type(elements)(**values)
        else:
          values = [
              (name, _to_representative_value(field_type, elements[name]))
              for name, field_type in field_types
          ]
          return collections.OrderedDict(values)
      else:
        return tuple(
            _to_representative_value(t, e) for t, e in zip(type_spec, elements)
        )
    else:
      raise ValueError(
          'Coercing a dataset with elements of expected type {!s}, '
          'produced a value with incompatible type `{!s}. Value: '
          '{!s}'.format(type_spec, type(elements), elements)
      )

  # tf.data.Dataset of tuples will unwrap the tuple in the `map()` call, so we
  # must pass a function taking *args. However, if the call was originally only
  # a single tuple, it is now "double wrapped" and must be unwrapped before
  # traversing.
  def _unwrap_args(*args):
    if len(args) == 1:
      return _to_representative_value(element_type, args[0])
    else:
      return _to_representative_value(element_type, args)

  return dataset.map(_unwrap_args)


def _uniquify_shared_names_with_suffix(
    graph_def: tf.compat.v1.GraphDef, suffix: str
) -> tf.compat.v1.GraphDef:
  """Appends unique identifier to any shared names present in `graph`."""
  # TODO: b/117428091 - Upgrade our TF serialization mechanisms in order to
  # unblock using more modern TF compositional constructs, and avoid direct
  # proto manipulation as is happening here.
  num_empty_shared_names = 0
  for node in graph_def.node:
    shared_name = node.attr.get('shared_name')
    if shared_name is not None:
      if not shared_name.s:
        # Encountered an empty string shared name, avoid creating a shared name
        # that starts with an underscore (not allowed by TF).
        shared_name.s = f'empty_{num_empty_shared_names}'.encode('utf-8')
        num_empty_shared_names += 1
      shared_name.s += b'_' + suffix.encode('utf-8')
  return graph_def


def deserialize_and_call_tf_computation(
    computation_proto: pb.Computation,
    arg: Optional[object],
    graph: tf.Graph,
    shared_names_suffix: str,
    session_token_tensor: tf.Tensor,
) -> tuple[Optional[str], object]:
  """Deserializes a TF computation and inserts it into `graph`.

  This method performs an action that can be considered roughly the opposite of
  what `tensorflow_serialization.serialize_py_fn_as_tf_computation` does. At
  the moment, it simply imports the graph in the current context. A future
  implementation may rely on different mechanisms. The caller should not be
  concerned with the specifics of the implementation. At this point, the method
  is expected to only be used within the body of another TF computation (within
  an instance of `tensorflow_computation_context.TensorFlowComputationContext`
  at the top of the stack), and potentially also in certain types of interpreted
  execution contexts (TBD).

  Args:
    computation_proto: An instance of `pb.Computation` with the `computation`
      one of equal to `tensorflow` to be deserialized and called.
    arg: The argument to invoke the computation with, or None if the computation
      does not specify a parameter type and does not expects one.
    graph: The graph to stamp into.
    shared_names_suffix: A string suffix to append to shared names within the
      graph. This must be unique for each call to
      `deserialize_and_call_tf_computation` for a given `graph`.
    session_token_tensor: A string tensor containing a token which has been fed
      a unique identifier for the current TensorFlow session.

  Returns:
    A tuple (init_op, result) where:
       init_op:  String name of an op to initialize the graph.
       result: The results to be fetched from TensorFlow. Depending on
           the type of the result, this can be `tf.Tensor` or `tf.data.Dataset`
           instances, or a nested structure (such as an
           `structure.Struct`).

  Raises:
    TypeError: If the arguments are of the wrong types.
    ValueError: If `computation_proto` is not a TensorFlow computation proto.
  """
  py_typecheck.check_type(computation_proto, pb.Computation)
  computation_oneof = computation_proto.WhichOneof('computation')
  if computation_oneof != 'tensorflow':
    raise ValueError(
        'Expected a TensorFlow computation, got {}.'.format(computation_oneof)
    )
  py_typecheck.check_type(graph, tf.Graph)
  with graph.as_default():
    type_spec = federated_language.FunctionType.from_proto(
        computation_proto.type
    )
    if type_spec.parameter is None:
      if arg is None:
        input_map = {}
      else:
        raise TypeError(
            'The computation declared no parameters; encountered an unexpected '
            'argument {}.'.format(arg)
        )
    elif arg is None:
      raise TypeError(
          'The computation declared a parameter of type {}, but the argument '
          'was not supplied.'.format(
              type_spec.parameter,
          )
      )
    else:
      arg_type, arg_binding = capture_result_from_graph(arg, graph)
      if not type_spec.parameter.is_assignable_from(arg_type):
        raise TypeError(
            'The computation declared a parameter of type {}, but the argument '
            'is of a mismatching type {}.'.format(
                type_spec.parameter,
                arg_type,
            )
        )
      else:
        input_map = {
            k: graph.get_tensor_by_name(v)
            for k, v in _compute_map_from_bindings(
                computation_proto.tensorflow.parameter, arg_binding
            ).items()
        }
    # Add potential control dep remappings.
    if input_map:
      input_map = graph_utils.add_control_dep_mappings(input_map)
    session_token_tensor_name = (
        computation_proto.tensorflow.session_token_tensor_name
    )
    if session_token_tensor_name:
      input_map[session_token_tensor_name] = session_token_tensor
    return_elements = extract_tensor_names_from_binding(
        computation_proto.tensorflow.result
    )
    orig_init_op_name = computation_proto.tensorflow.initialize_op
    if orig_init_op_name:
      return_elements.append(orig_init_op_name)

    graph_def = serialization_utils.unpack_graph_def(
        computation_proto.tensorflow.graph_def
    )
    graph_def = _uniquify_shared_names_with_suffix(
        graph_def, shared_names_suffix
    )
    # Note: Unlike MetaGraphDef, the GraphDef alone contains no information
    # about collections, and hence, when we import a graph with Variables,
    # those Variables are not added to global collections, and hence
    # functions like tf.compat.v1.global_variables_initializers() will not
    # contain their initialization ops.
    output_tensors = tf.graph_util.import_graph_def(
        graph_def,
        input_map,
        return_elements,
        # Note: It is very important not to return any names from the original
        # computation_proto.tensorflow.graph_def, those names might or might not
        # be valid in the current graph. Using a different scope makes the graph
        # somewhat more readable, since _N style de-duplication of graph
        # node names is less likely to be needed.
        name='subcomputation',
    )

    output_map = {k: v for k, v in zip(return_elements, output_tensors)}
    new_init_op_name = output_map.pop(orig_init_op_name, None)
    return (
        new_init_op_name,
        _assemble_result_from_graph(
            type_spec.result,
            computation_proto.tensorflow.result,
            output_map,
        ),
    )
