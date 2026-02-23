"""Main script to convert LiteRT models to the SR format"""

import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import datetime
import glob
import csv
import re
from jinja2 import Environment, FileSystemLoader

# import platform
from .gen_model_cpp import generate_model_cpp
from .gen_input_expected_data import generate_input_expected_data
from .generate_micro_mutable_op_resolver_from_model import (
    generate_micro_mutable_ops_resolver_header,
)
from .utils import get_platform_path


# Function to expand wildcards in input paths
def expand_wildcards(file_paths):
    """expand wildcards"""

    expanded_paths = []
    for path in file_paths:
        # Check if the path contains a wildcard
        if "*" in path:
            # Expand the wildcard to actual file names and sort them
            expanded_file_paths = sorted(glob.glob(path))
            # Extend the list with the sorted paths
            expanded_paths.extend(expanded_file_paths)
        else:
            # If no wildcard, add the path as is
            expanded_paths.append(path)
    return expanded_paths


def gen_model_script(new_model_file, args, env, license_header):
    """Generate the model script outputs"""

    if "flash" in args.system_config:
        weights_loc = "flash"
    else:
        weights_loc = "sram"

    # Generate model C++ code
    generate_model_cpp(
        new_model_file,
        args.output_dir,
        args.model_file_out,
        weights_loc,
        args.arena_cache_size,
        args.model_namespace,
        env,
        license_header,
    )

    # Generate micro mutable op resolver code
    common_path = os.path.dirname(new_model_file)
    if common_path == "":
        common_path = "."
    generate_micro_mutable_ops_resolver_header(
        common_path,
        [os.path.basename(new_model_file)],
        args.output_dir,
        args.model_namespace,
        license_header,
    )

    # Open the source file in read mode and the destination file in append mode
    src_fn = get_platform_path(
        args.output_dir + "/" + args.model_namespace + "_micro_mutable_op_resolver.hpp"
    )
    dest_fn = get_platform_path(args.output_dir + "/" + args.model_file_out + ".cc")
    with (
        open(src_fn, "r", encoding="utf-8") as source_file,
        open(dest_fn, "a", encoding="utf-8") as destination_file,
    ):
        # Read the content from the source file
        content = source_file.read()
        # Append the content to the destination file
        destination_file.write(content)

    # Generate on the original file
    generate_micro_mutable_ops_resolver_header(
        os.path.dirname(os.path.abspath(args.model_file)),
        [os.path.basename(args.model_file)],
        args.output_dir,
        "orig",
        license_header,
    )

    resolver_file = get_platform_path(
        args.output_dir + "/" + "orig_micro_mutable_op_resolver.hpp"
    )
    with open(resolver_file, "r", encoding="utf-8") as source_file:
        content = source_file.read()
        if "AddSynai" in content:
            synai_ethosu_op_found = 1
        elif "AddEthosU" in content:
            synai_ethosu_op_found = 2
        else:
            synai_ethosu_op_found = 0

    # Delete micro mutable op resolver file if it exists
    micro_mutable_file = get_platform_path(
        args.output_dir + "/" + args.model_file_out + "_micro_mutable_op_resolver.hpp"
    )
    if os.path.exists(micro_mutable_file):
        os.remove(micro_mutable_file)

    # Delete micro mutable op resolver file if it exists
    micro_mutable_file = get_platform_path(
        args.output_dir + "/" + "orig_micro_mutable_op_resolver.hpp"
    )
    if os.path.exists(micro_mutable_file):
        os.remove(micro_mutable_file)

    return synai_ethosu_op_found


def gen_inout_script(synai_ethosu_op_found, args, license_header):
    """Generate the inout script results"""

    # Check if AddSynai or AddEthosU is present in the contents of micro mutable op resolver
    if synai_ethosu_op_found > 0:
        if synai_ethosu_op_found == 1:
            print(
                "Synai custom op found in the model, skipping expected output generation"
            )
        else:
            print(
                "EthosU custom op found in the model, skipping expected output generation"
            )
    else:
        if args.input:
            generate_input_expected_data(
                args.model_file,
                args.output_dir,
                args.model_file_out,
                license_header,
                args.input,
            )
        else:
            generate_input_expected_data(
                args.model_file,
                args.output_dir,
                args.model_file_out,
                license_header,
            )


def setup_input(args):
    """Process inputs"""

    # Expand wildcards in input file paths
    if args.input:
        args.input = expand_wildcards(args.input)

    # Detect the model location
    if "weights_lpmem" in args.system_config:
        model_loc = "lpmem"
    elif "all_vmem" in args.system_config:
        model_loc = "vmem"
    else:
        model_loc = "flash"

    # Determine which scripts to run
    scripts_to_run = []
    if args.script:
        scripts_to_run = args.script
    else:
        scripts_to_run = ["model", "inout"]

    # Check if vela compilation is needed or not
    # If not that means the user is trying to run a non-vela model
    # In that case force the file name to no vela one
    file_name = os.path.basename(args.model_file)
    args.model_file = os.path.abspath(args.model_file)

    # Grab the summary file
    model_name = args.model_file.split("/")[-1].replace(".tflite", "")

    if args.compiler == "vela":
        new_tflite_file_name = file_name.split(".")[0] + "_vela.tflite"
    elif args.compiler == "synai":
        new_tflite_file_name = file_name.split(".")[0] + "_synai.tflite"
    elif args.compiler == "none":
        new_tflite_file_name = os.path.basename(args.model_file)
    else:
        new_tflite_file_name = os.path.basename(args.model_file)
        print("Invalid compiler option")
        sys.exit(1)

    new_model_file = get_platform_path(args.output_dir + "/" + new_tflite_file_name)

    return args, scripts_to_run, new_model_file, model_name, model_loc


def sr_get_compile_log(out_dir):
    """Get the Vela log text"""

    # Get the logs
    logfiles = glob.glob(f"{out_dir}/*vela.log")

    # return ext
    log_text = ""
    for logfile in logfiles:
        with open(logfile, "r", encoding="utf-8") as f:
            log_text = f.read()

    return log_text


def get_vela_summary(summary_file):
    """
    Parses a CSV file into a list of dictionaries, where each dictionary
    represents a row and uses the header row as keys.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        list: A list of dictionaries, or an empty list if the file is not found.
    """

    data = []
    try:
        with open(summary_file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"Error: The file '{summary_file}' was not found.")

    if len(data) == 1:
        data = data[0]
    for key in data.keys():
        print(f"{key} = {data[key]}")
    return data


def sr_check_model(results_dict):
    """Check model on SR data file to see if it fits"""

    # check for a bad compile
    if results_dict is None:
        return False, None
    if results_dict["cycles_npu"] == 0:
        return False, results_dict

    # Get the success story
    success = True

    # Setup the default performance data
    core_clock = int(float(results_dict["core_clock"]))
    perf_data = {
        "core_clock": core_clock,
        "cycles_npu": 0,
        "inferences_per_sec": 0,
        "inference_time": 0,
        "weights_size": 0,
        "arena_cache_size": 0,
        "vmem_size": 0,
        "lpmem_size": 0,
        "flash_size": 0,
        "vmem_size_limit": results_dict["vmem_size_limit"],
        "lpmem_size_limit": results_dict["lpmem_size_limit"],
        "model_loc": results_dict["model_loc"],
        "system_config": results_dict["system_config"],
        "vela_log": results_dict["vela_log"],
    }

    # Update performance data
    cycles_npu = int(float(results_dict["cycles_npu"]))
    inferences_per_sec = float(results_dict["inferences_per_second"])
    inference_time = float(results_dict["inference_time"])

    perf_data["cycles_npu"] = cycles_npu
    perf_data["inferences_per_sec"] = inferences_per_sec
    perf_data["inference_time"] = inference_time

    perf_data["weights_size"] = int(
        float(results_dict["off_chip_flash_memory_used"]) * 1024
    )
    perf_data["arena_cache_size"] = int(float(results_dict["arena_cache_size"]) * 1024)

    # Check the mode
    if results_dict["model_loc"] == "vmem":
        perf_data["vmem_size"] = (
            perf_data["weights_size"] + perf_data["arena_cache_size"]
        )
    elif results_dict["model_loc"] == "lpmem":
        perf_data["vmem_size"] = perf_data["arena_cache_size"]
        perf_data["lpmem_size"] = perf_data["weights_size"]
    else:
        perf_data["vmem_size"] = perf_data["arena_cache_size"]
        perf_data["flash_size"] = perf_data["weights_size"]

    # Check memory limits
    if perf_data["vmem_size"] > results_dict["vmem_size_limit"]:
        success = False
    if perf_data["lpmem_size"] > results_dict["lpmem_size_limit"]:
        success = False

    return success, perf_data


def run_vela(args):
    """Run the vela compiler"""

    # get the types of models
    model_types, _ = get_model_types(args.system_config_ini_file)

    # Get the INI file, either override or internal files
    if args.system_config_ini_file:
        arm_config = args.system_config_ini_file
    else:
        arm_config = model_types[args.system_config][0]

    # Set the memory mode
    if args.memory_mode:
        memory_mode = f"--memory-mode={args.memory_mode}"
    else:
        memory_mode = f"--memory-mode={model_types[args.system_config][1][-1]}"

    # Generate vela optimized model
    vela_params = [
        "vela",
        "--output-dir",
        args.output_dir,
        f"--accelerator-config={args.accel_config}",
        "--optimise=" + args.optimize,
        f"--config={arm_config}",
        memory_mode,
        f"--system-config={args.system_config}",
    ]
    if args.arena_cache_size:
        vela_params.append(f"--arena-cache-size={args.arena_cache_size}")
    if args.verbose_cycle_estimate:
        vela_params.append("--verbose-cycle-estimate")
    if args.verbose_all:
        vela_params.append("--verbose-all")
    vela_params.append(args.model_file)

    print("************ VELA ************")
    vela_log = ""
    try:
        vela_result = subprocess.run(vela_params, capture_output=True, check=True)
        vela_log += vela_result.stdout.decode("utf-8")
        vela_log += "\n"
        vela_log += vela_result.stderr.decode("utf-8")

        # Grab the summary file
        model_name = args.model_file.split("/")[-1].replace(".tflite", "")
        summary_file = (
            f"{args.output_dir}/{model_name}_summary_{args.system_config}.csv"
        )
        results = get_vela_summary(summary_file)
        results["vmem_size_limit"] = args.vmem_size_limit
        results["lpmem_size_limit"] = args.lpmem_size_limit

    except subprocess.CalledProcessError as e:
        print("Compilation failed:")
        results = {"cycles_npu": 0}
        vela_log += e.stdout.decode("utf-8")
        vela_log += "\n"
        vela_log += e.stderr.decode("utf-8")

    # print the log
    results["vela_log"] = vela_log
    print(vela_log)

    # Store the logs as well
    with open(f"{args.output_dir}/{model_name}_vela.log", "w", encoding="utf-8") as fp:
        fp.write(vela_log)
    print("********* END OF VELA *********")

    return results


def compiler_main(args):  # pylint: disable=R0914
    """Main function with input args"""

    # Creating a temporary directory if output dir is not provided
    tmp_dir = None
    if args.output_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732
        args.output_dir = tmp_dir.name

    results = None
    synai_ethosu_op_found = 0
    args, scripts_to_run, new_model_file, _, model_loc = setup_input(args)

    # Get the path to the directory containing this script
    script_dir = Path(__file__).parent
    get_model_types()

    print(f"script_dir = {script_dir}")
    files = glob.glob(f"{script_dir}/*")
    for file in files:
        print(f"file {file}")

    # Initialize Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(script_dir / "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    header_template = env.get_template("header_template.txt")
    license_header = header_template.render(
        script_name=script_dir.name,
        file_name=Path(args.model_file).name,
        gen_time=datetime.datetime.now(),
        year=datetime.datetime.now().year,
    )

    if args.compiler == "vela":
        results = run_vela(args)
        results["model_loc"] = model_loc
    elif args.compiler == "synai":
        # Generate synai optimized model
        print("*********** SYNAI **********")
        synai_params = [
            "synai",
            "--output-dir",
            os.path.dirname(args.model_file),
            args.model_file,
        ]
        subprocess.run(synai_params, check=True)
        print("******** END OF SYNAI ********")
    else:
        print("******* No Compilation *******")

    # Run the selected scripts if it compiled
    if results["cycles_npu"]:
        for script in scripts_to_run:
            if script == "model":
                synai_ethosu_op_found = gen_model_script(
                    new_model_file, args, env, license_header
                )
            elif script == "inout":
                gen_inout_script(synai_ethosu_op_found, args, license_header)

    # Cleaning up the temporary directory if it was created
    if tmp_dir:
        tmp_dir.cleanup()

    return results


def get_argparse_defaults(parser: argparse.ArgumentParser) -> dict:
    """
    Return a dictionary of all argparse defaults for the given parser.
    """
    return {
        action.dest: action.default
        for action in parser._actions  # pylint: disable=W0212
        if action.dest != "help"
    }


def get_args_from_call(parser, **kwargs):
    """get kwargs and merges with default args"""

    # Get default args
    arg_defaults = get_argparse_defaults(parser)

    # Update inputs with defaults
    for key in arg_defaults.keys():
        if key not in kwargs:
            kwargs[key] = arg_defaults[key]

    args = argparse.Namespace(**kwargs)
    return args


def sr_model_compiler(**kwargs):
    """Python entry functions for the call"""

    # Get default args
    parser = get_compiler_argparser()
    args = get_args_from_call(parser=parser, **kwargs)
    return compiler_main(args)


def get_model_types(ini_file_path=None):
    """Reports available models from ini files"""

    # Setup variables
    if ini_file_path:
        ini_files = [ini_file_path]
    else:
        # Grab all the ini files
        script_dir = Path(__file__).parent
        ini_files = glob.glob(f"{script_dir}/config/*.ini")

    model_types = {}
    full_memory_list = []

    for ini_file in ini_files:
        config_list = []
        memory_list = []

        with open(ini_file, "r", encoding="utf-8") as fp:
            lines = fp.readlines()

            for line in lines:
                s = re.search(r"\[System_Config\.([^\]]+)\]", line)
                m = re.search(r"\[Memory_Mode\.([^\]]+)\]", line)
                if s:
                    config_list.append(s.group(1))
                if m:
                    memory_list.append(m.group(1))

            # Fill the arrays
            for config in config_list:
                model_types[config] = (ini_file, memory_list)
            for mem in memory_list:
                if not mem in full_memory_list:
                    full_memory_list.append(mem)

    return (model_types, full_memory_list)


def print_modes(args):
    """Print the available modes"""

    model_types, memory_modes = get_model_types(args.system_config_ini_file)
    print("Analyzing available options for systems and memory")
    print("--system_config {")
    for model_type in model_types.keys():
        print(f"    {model_type},")
    print("}")

    print("--memory-modes {")
    for memory_mode in memory_modes:
        print(f"    {memory_mode},")
    print("}")


def get_compiler_argparser():
    """Parse command line arguments"""

    # get the types of models
    model_types, memory_modes = get_model_types()

    # Define args
    parser = argparse.ArgumentParser(
        description="Wrapper script to compile a TFLite model onto SR devices."
    )
    parser.add_argument(
        "-m", "--model-file", type=str, help="Path to TFLite model file"
    )
    parser.add_argument(
        "--get-modes",
        action="store_true",
        help="Report all the available system and memory modes",
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        choices=memory_modes,
        help="Sets memory architecture",
    )
    parser.add_argument(
        "--system-config-ini-file",
        type=str,
        help="Points to a config ini file",
    )
    parser.add_argument(
        "--system-config",
        type=str,
        default="sr100_npu_400MHz_all_vmem",
        choices=list(model_types.keys()),
        help="Sets system config selection",
    )
    parser.add_argument(
        "--accel-config",
        type=str,
        default="ethos-u55-128",
        help="Sets NPU size and version",
    )
    parser.add_argument(
        "--vmem-size-limit",
        type=int,
        default=1536000,
        help="Sets limit for vmem",
    )
    parser.add_argument(
        "--lpmem-size-limit",
        type=int,
        default=1536000,
        help="Sets limit for lpmem (operates at 1/4 speed of vmem)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Directory to output generated files",
    )
    parser.add_argument(
        "--model-namespace",
        type=str,
        help="Sets the model namespace",
        default="model",
    )
    parser.add_argument(
        "-n",
        "--model-file-out",
        type=str,
        help="Name of the output cc file for the model",
        default="model",
    )
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        nargs="+",
        choices=["model", "inout"],
        default=["model"],
        help="Choose specific scripts to run, if not provided then run all scripts",
    )
    parser.add_argument(
        "-i", "--input", type=str, nargs="+", help="List of input npy/bin files"
    )
    parser.add_argument(
        "-c",
        "--compiler",
        type=str,
        choices=["vela", "synai", "none"],
        help="Choose target compiler",
        default="vela",
    )
    parser.add_argument(
        "--arena-cache-size",
        type=int,
        default=1024000,
        help="Sets the model arena cache size in bytes",
    )
    parser.add_argument(
        "-v",
        "--verbose-all",
        action="store_true",
        help="Turns on verbose all for the compiler",
    )
    parser.add_argument(
        "--verbose-cycle-estimate",
        action="store_true",
        help="Turns on verbose cycle estimation",
    )
    parser.add_argument(
        "-p",
        "--optimize",
        type=str,
        choices=["Performance", "Size"],
        help="Choose optimization Type",
        default="Size",
        required=False,
    )

    return parser


def main():
    """Main for the command line compiler"""
    parser = get_compiler_argparser()
    args = parser.parse_args()

    # get the types of models
    if args.get_modes:
        print_modes(args)
        return 0

    # Runs the vela compiler
    results = compiler_main(args)

    # Checks the SR100 mapping
    success, perf_data = sr_check_model(results)

    # Reports the
    if success:
        print(f"Successfully mapped {args.model_file} onto sr")
        returncode = 0
    else:
        print(f"ERROR:: Failed to map {args.model_file} onto sr")
        returncode = 1
    for key, value in perf_data.items():
        print(f"   {key} = {value}")

    return returncode


if __name__ == "__main__":
    main()
