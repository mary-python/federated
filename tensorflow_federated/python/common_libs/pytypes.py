## Copyright 2022, The TensorFlow Federated Authors.
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
"""Python type annotations aliases."""

from typing import OrderedDict, Sequence, Union

import numpy as np
import tensorflow as tf

# A type that acts like a tensor (possibly mutlidemensional or a scalar). This
# frequently is the input to a `tf.function` or `tff.tf_computation` decorated
# callable.
TensorLike = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor, tf.Variable,
                   np.ndarray, np.number, float, int, str, bytes]

# A potentially nested TensorLike object.
TensorStruct = Union[TensorLike, Sequence['TensorStruct'],
                     OrderedDict[str, 'TensorStruct']]

# A variant type covering any type of tensor spec from TensorFlow.
TensorSpecVariant = Union[tf.TensorSpec, tf.SparseTensorSpec,
                          tf.RaggedTensorSpec]

# A potentially nested TensorSpecVariant object.
TensorSpecStruct = Union[TensorSpecVariant, Sequence['TensorSpecStruct'],
                         OrderedDict[str, 'TensorSpecStruct']]
