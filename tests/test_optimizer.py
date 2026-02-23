#!/usr/bin/env python3
"""Testing different optimizers of models"""

import pytest
from sr_model_compiler import sr100_model_optimizer

model_test_list = [
    ("tests/models/hello_world/hello_world.tflite", 2048, 2048, "vmem", True),
    ("tests/models/hello_world/hello_world.tflite", 16, 2048, "lpmem", False),
    ("tests/models/hello_world/hello_world.tflite", 32, 1024, "flash", True),
    (
        "tests/models/uc_person_classification/person_classification_256x448.tflite",
        600000,
        1536000,
        "lpmem",
        True,
    ),
    (
        "tests/models/uc_person_classification/person_classification_448x640.tflite",
        1536000,
        536000,
        "flash",
        True,
    ),
    (
        "tests/models/uc_person_detection/person_detection_256x480.tflite",
        400000,
        1536000,
        "lpmem",
        False,
    ),
]


@pytest.mark.parametrize(
    "model_file, vmem_size_limit, lpmem_size_limit, model_loc, success_expect",
    model_test_list,
)
def test_model_optimizer(
    model_file, vmem_size_limit, lpmem_size_limit, model_loc, success_expect
):
    """builds a model and tests outputs"""

    # Get model name to build directory
    success, results = sr100_model_optimizer(
        model_file=model_file,
        vmem_size_limit=vmem_size_limit,
        lpmem_size_limit=lpmem_size_limit,
    )

    assert success == success_expect, f"Optimization failed for {model_file}"
    assert (
        results["model_loc"] == model_loc
    ), f'{model_file} - Expected model location {model_loc}, got {results["model_loc"]}'


if __name__ == "__main__":

    # Run all the tests and update if needed
    for model_test in model_test_list:
        (
            model_file_v,
            vmem_size_limit_v,
            lpmem_size_limit_v,
            model_loc_v,
            success_expect_v,
        ) = model_test
        test_model_optimizer(
            model_file_v,
            vmem_size_limit_v,
            lpmem_size_limit_v,
            model_loc_v,
            success_expect_v,
        )
