# Copyright 2020, The TensorFlow Federated Authors.
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
"""Defines a template for a stateful process that aggregates values."""

import federated_language
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.templates import errors
from tensorflow_federated.python.core.templates import measured_process

# Index of the argument to next_fn representing value to be aggregated.
_INPUT_PARAM_INDEX = 1


class AggregationNotFederatedError(TypeError):
  """`TypeError` for aggregation functions not being federated."""


class AggregationPlacementError(TypeError):
  """`TypeError` for aggregation types not being placed as expected."""


class AggregationProcess(measured_process.MeasuredProcess):
  """A stateful process that aggregates values.

  This class inherits the constraints documented by
  `tff.templates.MeasuredProcess`.

  A `tff.templates.AggregationProcess` is a `tff.templates.MeasuredProcess`
  that formalizes the type signature of `initialize_fn` and `next_fn` for
  aggregation.

  Compared to the `tff.templates.MeasuredProcess`, this class requires a second
  input argument, which is a value placed at `CLIENTS` and to be aggregated.
  The `result` field of returned `tff.templates.MeasuredProcessOutput`,
  representing the aggregate, must be placed at `SERVER` and does not
  necessarily need to have type signature equal to the type signature of the
  second input argument.

  The intended composition pattern for `tff.templates.AggregationProcess` is
  that of nesting. An aggregation will broadly consist of three logical parts:
    - A pre-aggregation computation placed at `CLIENTS`.
    - Actual aggregation.
    - A post-aggregation computation placed at `SERVER`.
  The second step can be realized by direct application of appropriate intrinsic
  such as `federated_language.federated_sum`, or by delegation to (one or more)
  "inner"
  aggregation processes.

  Both `initialize` and `next` must be `federated_language.Computation`s with
  the following
  type signatures:
    - initialize: `( -> S@SERVER)`
    - next: `(<S@SERVER, V@CLIENTS, *> ->
              <state=S@SERVER, result=V'@SERVER, measurements=M@SERVER>)`
  where `*` represents optional other arguments placed at `CLIENTS`. This can be
  used for weighted aggregation, where the third parameter is the weight.

  Note that while the value type to be aggregated will often be preserved
  (i.e., `V == V'`), it is not required. An example is sampling-based
  aggregation.
  """

  def __init__(
      self,
      initialize_fn: federated_language.framework.Computation,
      next_fn: federated_language.framework.Computation,
  ):
    """Creates a `tff.templates.AggregationProcess`.

    Args:
      initialize_fn: A no-arg `federated_language.Computation` that returns the
        initial state of the aggregation process. The returned state must be a
        server-placed federated value. Let the type of this state be called
        `S@SERVER`.
      next_fn: A `federated_language.Computation` that represents the iterated
        function. `next_fn` must accept at least two arguments, the first of
        which is of a type assignable from the state type `S@SERVER` and the
        second of which is client-placed data of type `V@CLIENTS`. `next_fn`
        must return a `MeasuredProcessOutput` where the `state` attribute is
        assignable to the first argument and the `result` is value placed at
        `SERVER`.

    Raises:
      TypeError: If `initialize_fn` and `next_fn` are not instances of
        `federated_language.Computation`.
      TemplateInitFnParamNotEmptyError: If `initialize_fn` has any input
        arguments.
      TemplateStateNotAssignableError: If the `state` returned by either
        `initialize_fn` or `next_fn` is not assignable to the first input
        argument of `next_fn`.
      TemplateNotMeasuredProcessOutputError: If `next_fn` does not return a
        `MeasuredProcessOutput`.
      TemplateNextFnNumArgsError: If `next_fn` does not have at least two
        input arguments.
      AggregationNotFederatedError: If `initialize_fn` and `next_fn` are not
        computations operating on federated types.
      AggregationPlacementError: If the placements of `initialize_fn` and
        `next_fn` are not matching the expected type signature.
    """
    # Calling super class __init__ first ensures that
    # next_fn.type_signature.result is a `MeasuredProcessOutput`, make our
    # validation here easier as that must be true.
    super().__init__(initialize_fn, next_fn, next_is_multi_arg=True)

    if not isinstance(
        initialize_fn.type_signature.result, federated_language.FederatedType
    ):
      raise AggregationNotFederatedError(
          'Provided `initialize_fn` must return a federated type, but found '
          f'return type:\n{initialize_fn.type_signature.result}\nTip: If you '
          'see a collection of federated types, try wrapping the returned '
          'value in `federated_language.federated_zip` before returning.'
      )
    next_types = structure.flatten(
        next_fn.type_signature.parameter
    ) + structure.flatten(next_fn.type_signature.result)
    non_federated_types = [
        t
        for t in next_types
        if not isinstance(t, federated_language.FederatedType)
    ]
    if non_federated_types:
      offending_types_str = '\n- '.join(str(t) for t in non_federated_types)
      raise AggregationNotFederatedError(
          'Provided `next_fn` must both be a *federated* computations, that '
          'is, operate on `federated_language.FederatedType`s, but found\n'
          f'next_fn with type signature:\n{next_fn.type_signature}\n'
          f'The non-federated types are:\n {offending_types_str}.'
      )

    if (
        initialize_fn.type_signature.result.placement
        != federated_language.SERVER
    ):
      raise AggregationPlacementError(
          'The state controlled by an `AggregationProcess` must be placed at '
          f'the SERVER, but found type: {initialize_fn.type_signature.result}.'
      )
    # Note that state of next_fn being placed at SERVER is now ensured by the
    # assertions in base class which would otherwise raise
    # errors.TemplateStateNotAssignableError.

    next_fn_param = next_fn.type_signature.parameter
    next_fn_result = next_fn.type_signature.result
    if len(next_fn_param) < 2:
      raise errors.TemplateNextFnNumArgsError(
          'The `next_fn` must have at least two input arguments, but found '
          f'the following input type: {next_fn_param}.'
      )

    if (
        next_fn_param[_INPUT_PARAM_INDEX].placement
        != federated_language.CLIENTS
    ):
      raise AggregationPlacementError(
          'The second input argument of `next_fn` must be placed at CLIENTS '
          f'but found {next_fn_param[_INPUT_PARAM_INDEX]}.'
      )

    if next_fn_result.result.placement != federated_language.SERVER:
      raise AggregationPlacementError(
          'The "result" attribute of return type of `next_fn` must be placed '
          f'at SERVER, but found {next_fn_result.result}.'
      )
    if next_fn_result.measurements.placement != federated_language.SERVER:
      raise AggregationPlacementError(
          'The "measurements" attribute of return type of `next_fn` must be '
          f'placed at SERVER, but found {next_fn_result.measurements}.'
      )

  @property
  def next(self) -> federated_language.framework.Computation:
    """A `federated_language.Computation` that runs one iteration of the process.

    Its first argument should always be the current state (originally produced
    by the `initialize` attribute), the second argument must be the input placed
    at `CLIENTS`, and the return type must be a
    `tff.templates.MeasuredProcessOutput` with each field placed at `SERVER`.

    Returns:
      A `federated_language.Computation`.
    """
    return super().next

  @property
  def is_weighted(self) -> bool:
    """True if `next` takes a third argument for weights."""
    return len(self.next.type_signature.parameter) == 3  # pytype: disable=wrong-arg-types
