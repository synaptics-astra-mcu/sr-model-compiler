"""Main script to optimize a SR110 model"""

import argparse
import tempfile
from .sr_model_compiler import (
    sr_model_compiler,
    sr_check_model,
    get_args_from_call,
)


def model_optimizer_search(args):
    """Searches for the model that fits"""

    # Using TemporaryDirectory as a context manager for automatic cleanup
    results = None
    with tempfile.TemporaryDirectory() as tmpdirname:

        # You can perform operations within the temporary directory
        output_dir = f"{tmpdirname}"

        # Gets minimum arena cache size
        results_size = sr_model_compiler(
            model_file=args.model_file, arena_cache_size=3072000, output_dir=output_dir
        )
        # Analyze the results
        weights_size = int(float(results_size["off_chip_flash_memory_used"]) * 1024)
        cache_size = int(float(results_size["sram_memory_used"]) * 1024)
        total_size = cache_size + weights_size

        # Determine the system configuration
        if total_size <= args.vmem_size_limit:
            system_config = "sr100_npu_400MHz_all_vmem"
            cache_size_increase = args.vmem_size_limit - total_size
        elif weights_size <= args.lpmem_size_limit:
            system_config = "sr100_npu_400MHz_tensor_vmem_weights_lpmem"
            cache_size_increase = args.vmem_size_limit - cache_size
        else:
            system_config = "sr100_npu_400MHz_tensor_vmem_weights_flash66MHz"
            cache_size_increase = args.vmem_size_limit - cache_size

        # Increase performance to vmem max
        if args.optimize == "Performance":
            cache_size += cache_size_increase

        # Run the final results
        results = sr_model_compiler(
            model_file=args.model_file,
            arena_cache_size=cache_size,
            system_config=system_config,
            output_dir=output_dir,
            vmem_size_limit=args.vmem_size_limit,
            lpmem_size_limit=args.lpmem_size_limit,
            optimize=args.optimize,
        )

    # Checks the SR100 mapping
    success, perf_data = sr_check_model(results)

    return success, perf_data


def sr100_model_optimizer(**kwargs):
    """Python entry functions for the call"""

    # Get default args
    parser = get_optimizer_argparser()
    args = get_args_from_call(parser, **kwargs)
    print(args)
    return model_optimizer_search(args)


def get_optimizer_argparser():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(
        description="Optimize memory location for a TFLite model for an SR100 devices."
    )
    parser.add_argument(
        "-m", "--model-file", type=str, help="Path to TFLite model file", required=True
    )
    parser.add_argument(
        "--vmem-size-limit", type=int, default=1536000, help="Set vmem size limit"
    )
    parser.add_argument(
        "--lpmem-size-limit", type=int, default=1536000, help="Set lpmem size limit"
    )
    parser.add_argument(
        "-p",
        "--optimize",
        type=str,
        default="Size",
        choices=["Performance", "Size"],
        help="Choose optimization Type",
    )
    return parser


def main():
    """Main for the command line compiler"""
    parser = get_optimizer_argparser()
    args = parser.parse_args()

    # Checks the SR100 mapping
    success, perf_data = model_optimizer_search(args)

    # Print performance data
    for key, value in perf_data.items():
        print(f"{key}: {value}")

    # Fine tune the model
    if success:
        returncode = 0
    else:
        returncode = 1
    return returncode


if __name__ == "__main__":
    main()
