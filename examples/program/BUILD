load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "computations",
    srcs = ["computations.py"],
    deps = [
        "//tensorflow_federated",
        "@federated_language//federated_language",
    ],
)

py_library(
    name = "program_logic",
    srcs = ["program_logic.py"],
    deps = ["@federated_language//federated_language"],
)

py_binary(
    name = "program",
    srcs = ["program.py"],
    deps = [
        ":computations",
        ":program_logic",
        "//tensorflow_federated",
        "@federated_language//federated_language",
    ],
)

py_test(
    name = "program_logic_test",
    srcs = ["program_logic_test.py"],
    deps = [
        ":computations",
        ":program_logic",
        "//tensorflow_federated",
        "@federated_language//federated_language",
    ],
)
