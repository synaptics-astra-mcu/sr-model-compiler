import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import binascii
import platform

# Define the choices and corresponding strings for tflite location
loc_choices = {
    "sram": "MODEL_TFLITE_ATTRIBUTE",  # SRAM
    "flash": "MODEL_TFLITE_ATTRIBUTE_FLASH",  # QSPI FLASH
}


def generate_model_cpp(
    tflite_path,
    output_dir,
    model_file,
    tflite_loc,
    arena_cache_size,
    namespace,
    env,
    license_header,
):
    """Generates a C++ source file that contains the TFLite model as a byte array."""

    tflite_loc_choice = loc_choices.get(tflite_loc, "MODEL_TFLITE_ATTRIBUTE")

    # Check for the vela output
    if not Path(tflite_path).is_file():
        raise Exception(f"{tflite_path} not found")

    # Resolve relative paths for output_dir and cpp_filename
    output_dir = Path(output_dir).resolve()
    cpp_filename = (output_dir / (model_file + ".cc")).resolve()
    if platform.system() == "Windows":
        print(
            f"++ Converting {Path(tflite_path).name} to {output_dir}\\{cpp_filename.name}"
        )
    else:
        print(
            f"++ Converting {Path(tflite_path).name} to {output_dir}/{cpp_filename.name}"
        )

    output_dir.mkdir(exist_ok=True)

    model_data, model_length = get_tflite_data(tflite_path)
    env.get_template("tflite.cc.template").stream(
        common_template_header=license_header,
        arena_cache_size=arena_cache_size,
        tflite_loc=tflite_loc,
        model_data=model_data,
        model_length=model_length,
        namespace=namespace,
        tflite_attribute=tflite_loc_choice,
    ).dump(str(cpp_filename))

    # Write the binary file
    flash_file = tflite_path.replace("_vela.tflite", ".bin")

    # Read vela bytes
    with open(tflite_path, "rb") as tflite_model:
        data = tflite_model.read()
    # Write to binary file
    with open(flash_file, "wb") as fp:
        fp.write(data)


def get_tflite_data(tflite_path):
    """
    Reads a binary file and returns a C style array as a
    list of strings.

    Argument:
        tflite_path:    path to the tflite model.

    Returns:
        tuple: (list of strings, int)
            - List of strings representing the C style array
            - Number of bytes in the binary file
    """
    with open(tflite_path, "rb") as tflite_model:
        data = tflite_model.read()

    bytes_per_line = 32
    hex_digits_per_line = bytes_per_line * 2
    hexstream = binascii.hexlify(data).decode("utf-8")
    hexstring = "{"

    for i in range(0, len(hexstream), 2):
        if 0 == (i % hex_digits_per_line):
            hexstring += "\n"
        hexstring += "0x" + hexstream[i : i + 2] + ", "

    hexstring += "};\n"
    return [hexstring], len(data)


# Optionally, you can still keep the command-line interface for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite_path", help="Model (.tflite) path", required=True)
    parser.add_argument("--output_dir", help="Output directory", required=True)
    parser.add_argument("-e", "--expression", action="append", default=[], dest="expr")
    parser.add_argument("--header", action="append", default=[], dest="headers")
    parser.add_argument(
        "-ns", "--namespaces", action="append", default=[], dest="namespaces"
    )
    parser.add_argument(
        "--license_template",
        type=str,
        help="Header template file",
        default="header_template.txt",
    )
    parser.add_argument(
        "-tl",
        "--tflite_loc",
        type=str,
        choices=loc_choices.keys(),
        help="Choose an option (sram, flash)",
        default="sram",
        required=False,
    )

    args = parser.parse_args()
    license_header = ""
    generate_model_cpp(
        args.tflite_path,
        args.output_dir,
        args.namespace,
        args.tflite_loc,
        license_header,
    )
