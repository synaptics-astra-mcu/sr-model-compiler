#!/usr/bin/env python3
"""Testing different builds of models"""

import os
import filecmp
import argparse
from pathlib import Path
import pytest
from sr_model_compiler import sr_model_compiler, call_shell_cmd

model_test_list = [
    (
        "tests/models/hello_world/hello_world.tflite",
        "sr100_npu_400MHz_all_vmem",
        "model",
    ),
    (
        "tests/models/uc_person_classification/person_classification_256x448.tflite",
        "sr100_npu_400MHz_all_vmem",
        "model_wqvga",
    ),
    (
        "tests/models/uc_person_classification/person_classification_448x640.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_lpmem",
        "model_vga",
    ),
    (
        "tests/models/uc_person_detection/person_detection_256x480.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_flash66MHz",
        "model_wqvga",
    ),
    (
        "tests/models/uc_person_detection/person_detection_480x640.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_flash100MHz",
        "model_vga",
    ),
    (
        "tests/models/uc_person_pose_detection/person_pose_detection_256x480.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_lpmem",
        "model_wqvga",
    ),
    (
        "tests/models/uc_person_pose_detection/person_pose_detection_480x640.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_lpmem",
        "model_vga",
    ),
    (
        "tests/models/uc_person_segmentation/person_segmentation_256x480.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_lpmem",
        "model_wqvga",
    ),
    (
        "tests/models/uc_person_segmentation/person_segmentation_480x640.tflite",
        "sr100_npu_400MHz_tensor_vmem_weights_lpmem",
        "model_vga",
    ),
]


def test_shell_cmd(tmp_path):
    """Test that python + shell command are the same outputs"""

    model, system_config, model_file_out = model_test_list[0]
    model_name = os.path.basename(model).replace(".tflite", "")

    # Building output directory
    out_dir = tmp_path

    # Run the shell command
    success, vela_log = call_shell_cmd(
        f"sr_model_compiler -m {model}"
        f" --output-dir {out_dir}"
        f" --system-config {system_config}"
        f" --model-file-out {model_file_out}"
    )
    assert success is True, f"Failed to run sr_model_compiler command: {vela_log}"
    # print(f"success = {success}")
    print(f"vela_log = {vela_log}")

    # Assert the model space file exists
    cc_file = f"{out_dir}/{model_file_out}.cc"
    assert os.path.exists(cc_file), f"Failed to find {cc_file}"

    # Read vela bytes
    flash_bin_golden_file = model.replace(".tflite", ".bin")
    flash_bin_file = f"{out_dir}/{model_name}.bin"

    # Compares the binary files
    assert filecmp.cmp(
        flash_bin_golden_file, flash_bin_file
    ), f"ERROR binfile mismathc {flash_bin_golden_file} with {flash_bin_file}"


@pytest.mark.parametrize(
    "model, system_config, model_file_out",
    model_test_list,
)
def test_model_compiler(
    tmp_path, model, system_config, model_file_out, update_bin_file=False
):
    """builds a model and tests outputs"""

    # Get model name to build directory
    model_name = os.path.basename(model).replace(".tflite", "")

    # Building output directory
    out_dir = f"{tmp_path}"

    # Run the comparison
    results = sr_model_compiler(
        model_file=model,
        output_dir=f"{out_dir}",
        system_config=system_config,
        model_file_out=model_file_out,
    )
    print(results)

    # Assert the model space file exists
    cc_file = f"{out_dir}/{model_file_out}.cc"
    assert os.path.exists(cc_file), f"Failed to find {cc_file}"

    # Read vela bytes
    flash_bin_golden_file = model.replace(".tflite", ".bin")
    flash_bin_file = f"{out_dir}/{model_name}.bin"

    # Temp to create vectors - ONLY USE IF UPDATED VECTORS
    if update_bin_file:
        with open(flash_bin_file, "rb") as tflite_model:
            data = tflite_model.read()
        with open(flash_bin_golden_file, "wb") as fp:
            fp.write(data)

    # Compares the binary files
    assert filecmp.cmp(
        flash_bin_golden_file, flash_bin_file
    ), f"ERROR binfile mismathc {flash_bin_golden_file} with {flash_bin_file}"


def test_float_model(tmp_path):
    """Tests a float model that should not map"""

    # Get model name to build directory
    model = "tests/models/hello_world/hello_world_float.tflite"
    model_name = "hello_world_float"
    model_dir = model_name

    # Building output directory
    out_dir = tmp_path / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)  #

    # Run the comparison
    results = sr_model_compiler(
        model_file=model, output_dir=f"{out_dir}", model_loc="sram"
    )
    cycles_npu = float(results["cycles_npu"])

    assert cycles_npu == 0.0, f"Failed to get 0 cycles in the NPU, found {cycles_npu}"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Wrapper script to compile a TFLite model onto SR devices."
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="tmp_build",
        help="Sets temporary build directory",
    )
    parser.add_argument(
        "--update",
        default=False,
        action="store_true",
        help="Updates the Golden test vectors",
    )
    args = parser.parse_args()

    # Run all the tests and update if needed
    for model_test in model_test_list:
        model_v, system_config_v, model_file_out_v = model_test
        test_model_compiler(
            Path(args.tmp_dir), model_v, system_config_v, model_file_out_v, args.update
        )

    # Test the float model as well
    test_float_model(Path(args.tmp_dir))
