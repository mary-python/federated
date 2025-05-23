load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":optimizers_packages",
        "//tensorflow_federated/python/learning:learning_users",
        "//tensorflow_federated/python/learning/algorithms:algorithms_packages",
        "//tensorflow_federated/python/learning/templates:templates_packages",
    ],
)

package_group(
    name = "optimizers_packages",
    packages = ["//tensorflow_federated/python/learning/optimizers/..."],
)

licenses(["notice"])

py_library(
    name = "adafactor",
    srcs = ["adafactor.py"],
    deps = [
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "adafactor_test",
    srcs = ["adafactor_test.py"],
    deps = [
        ":adafactor",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
    ],
)

py_library(
    name = "adagrad",
    srcs = ["adagrad.py"],
    deps = [
        ":nest_utils",
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "adagrad_test",
    srcs = ["adagrad_test.py"],
    deps = [
        ":adagrad",
        ":optimizer",
        ":optimizer_test_utils",
    ],
)

py_library(
    name = "adam",
    srcs = ["adam.py"],
    deps = [
        ":nest_utils",
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "adam_test",
    srcs = ["adam_test.py"],
    deps = [
        ":adam",
        ":optimizer",
        ":optimizer_test_utils",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    visibility = ["//tensorflow_federated/python/learning:__pkg__"],
    deps = [
        ":adafactor",
        ":adagrad",
        ":adam",
        ":adamw",
        ":optimizer",
        ":rmsprop",
        ":scheduling",
        ":sgdm",
        ":yogi",
    ],
)

py_library(
    name = "keras_optimizer",
    srcs = ["keras_optimizer.py"],
    deps = [":optimizer"],
)

py_test(
    name = "keras_optimizer_test",
    srcs = ["keras_optimizer_test.py"],
    deps = [
        ":keras_optimizer",
        ":optimizer",
        ":optimizer_test_utils",
        ":sgdm",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
    ],
)

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    deps = [
        ":adagrad",
        ":adam",
        ":rmsprop",
        ":scheduling",
        ":sgdm",
        ":yogi",
        "//tensorflow_federated/python/core/backends/native:execution_contexts",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_computation",
        "//tensorflow_federated/python/core/environments/tensorflow_frontend:tensorflow_types",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
)

py_test(
    name = "optimizer_test",
    srcs = ["optimizer_test.py"],
    deps = [":optimizer"],
)

py_library(
    name = "optimizer_test_utils",
    testonly = True,
    srcs = ["optimizer_test_utils.py"],
    deps = [":optimizer"],
)

py_test(
    name = "optimizer_test_utils_test",
    srcs = ["optimizer_test_utils_test.py"],
    deps = [":optimizer_test_utils"],
)

py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        ":nest_utils",
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "rmsprop_test",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":rmsprop",
    ],
)

py_library(
    name = "scheduling",
    srcs = ["scheduling.py"],
    deps = [":optimizer"],
)

py_test(
    name = "scheduling_test",
    srcs = ["scheduling_test.py"],
    deps = [
        ":adagrad",
        ":adam",
        ":optimizer",
        ":rmsprop",
        ":scheduling",
        ":sgdm",
        ":yogi",
    ],
)

py_library(
    name = "sgdm",
    srcs = ["sgdm.py"],
    deps = [
        ":nest_utils",
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_test(
    name = "sgdm_test",
    srcs = ["sgdm_test.py"],
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":sgdm",
    ],
)

py_library(
    name = "yogi",
    srcs = ["yogi.py"],
    deps = [
        ":nest_utils",
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_library(
    name = "adamw",
    srcs = ["adamw.py"],
    deps = [
        ":nest_utils",
        ":optimizer",
        "//tensorflow_federated/python/common_libs:structure",
    ],
)

py_library(
    name = "nest_utils",
    srcs = ["nest_utils.py"],
)

py_test(
    name = "yogi_test",
    srcs = ["yogi_test.py"],
    deps = [
        ":optimizer",
        ":optimizer_test_utils",
        ":yogi",
    ],
)

py_test(
    name = "adamw_test",
    srcs = ["adamw_test.py"],
    deps = [
        ":adamw",
        ":optimizer",
        ":optimizer_test_utils",
    ],
)

py_test(
    name = "nest_utils_test",
    srcs = ["nest_utils_test.py"],
    deps = [":nest_utils"],
)
