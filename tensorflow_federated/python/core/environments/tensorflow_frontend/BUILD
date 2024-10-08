load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":tensorflow_frontend_packages",
        "//tensorflow_federated/python/core/environments:environments_users",
        "//tensorflow_federated/python/core/environments/tensorflow:tensorflow_packages",
        "//tensorflow_federated/python/core/environments/tensorflow_backend:tensorflow_backend_packages",
    ],
)

package_group(
    name = "tensorflow_frontend_packages",
    packages = ["//tensorflow_federated/python/core/environments/tensorflow_frontend/..."],
)

licenses(["notice"])

py_library(
    name = "tensorflow_frontend",
    srcs = ["__init__.py"],
    visibility = ["//tools/python_package:python_package_tool"],
)

py_library(
    name = "tensorflow_computation",
    srcs = ["tensorflow_computation.py"],
    deps = [
        ":tensorflow_serialization",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/computation:computation_wrapper",
        "//tensorflow_federated/python/core/impl/computation:function_utils",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_analysis",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
    ],
)

py_test(
    name = "tensorflow_computation_test",
    size = "small",
    srcs = ["tensorflow_computation_test.py"],
    deps = [
        ":tensorflow_computation",
        "//tensorflow_federated/python/core/impl/computation:computation_wrapper",
        "//tensorflow_federated/python/core/impl/context_stack:get_context_stack",
        "//tensorflow_federated/python/core/impl/context_stack:runtime_error_context",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:placements",
        "//tensorflow_federated/python/core/impl/types:type_test_utils",
    ],
)

py_library(
    name = "tensorflow_computation_context",
    srcs = ["tensorflow_computation_context.py"],
    deps = [
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/common_libs:structure",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_base",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_conversions",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
    ],
)

py_test(
    name = "tensorflow_computation_context_test",
    size = "small",
    srcs = ["tensorflow_computation_context_test.py"],
    deps = [
        ":tensorflow_computation",
        ":tensorflow_computation_context",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/core/impl/computation:computation_impl",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
    ],
)

py_library(
    name = "tensorflow_serialization",
    srcs = ["tensorflow_serialization.py"],
    deps = [
        ":tensorflow_computation_context",
        ":variable_utils",
        "//tensorflow_federated/proto/v0:computation_py_pb2",
        "//tensorflow_federated/python/common_libs:py_typecheck",
        "//tensorflow_federated/python/core/impl/computation:computation_wrapper",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_base",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/core/impl/utils:tensorflow_utils",
        "//tensorflow_federated/python/tensorflow_libs:serialization_utils",
    ],
)

py_test(
    name = "tensorflow_serialization_test",
    size = "small",
    srcs = ["tensorflow_serialization_test.py"],
    deps = [
        ":tensorflow_serialization",
        "//tensorflow_federated/python/core/impl/context_stack:context_stack_impl",
        "//tensorflow_federated/python/core/impl/types:computation_types",
        "//tensorflow_federated/python/core/impl/types:type_serialization",
        "//tensorflow_federated/python/tensorflow_libs:serialization_utils",
        "//tensorflow_federated/python/tensorflow_libs:tensorflow_test_utils",
    ],
)

py_library(
    name = "variable_utils",
    srcs = ["variable_utils.py"],
)

py_test(
    name = "variable_utils_test",
    size = "small",
    timeout = "moderate",
    srcs = ["variable_utils_test.py"],
    deps = [":variable_utils"],
)
