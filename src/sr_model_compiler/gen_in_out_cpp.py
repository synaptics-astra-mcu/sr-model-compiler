import argparse
import glob
from mako.template import Template
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(
        description="Process input and output files for a template, including a namespace, allowing for different numbers of input and output files."
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        help="Input file(s) or wildcard pattern(s)",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--output",
        action="append",
        help="Output file(s) or wildcard pattern(s)",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--namespace",
        type=str,
        help="Namespace for the generated code",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Directory to save the generated 'model_io.cc' file",
        required=True,
    )

    args = parser.parse_args()

    print("++ Generating model_io.cc from input and output bin files")

    # Expand wildcards and collect all actual file paths
    input_files = expand_files(args.input)
    output_files = expand_files(args.output) if args.output else []

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the template relative to the script's directory
    template_path = os.path.join(script_dir, "templates", "io_template.mako")
    with open(template_path, "r") as f:
        template_content = f.read()
    template = Template(template_content)

    input_data_list = []
    input_data_size_list = []
    for f in input_files:
        data, size = read_file_data(f)
        input_data_list.append(data)
        input_data_size_list.append(size)

    output_data_list = []
    output_data_size_list = []
    for f in output_files:
        data, size = read_file_data(f)
        output_data_list.append(data)
        output_data_size_list.append(size)

    # Render the template with input and output files
    rendered = template.render(
        namespace=args.namespace,
        input_data_list=input_data_list,
        output_data_list=output_data_list,
        input_data_size_list=input_data_size_list,
        output_data_size_list=output_data_size_list,
    )

    # Save the rendered content to a new file
    output_file_path = os.path.join(args.output_dir, args.namespace + "_io.cc")
    with open(output_file_path, "w") as f:
        f.write(rendered)


def expand_files(patterns):
    """Expand wildcard patterns to actual file paths."""
    all_files = []
    for pattern in patterns:
        # Expand the wildcard pattern to a list of files
        matched_files = glob.glob(pattern)
        all_files.extend(matched_files)
    return all_files


def read_file_data(file_path):
    """Read binary file content, and format as C++ array initialization with 32 bytes per line."""
    with open(file_path, "rb") as f:
        # Read all bytes from the file and convert to a NumPy array
        bytes_data = np.fromfile(f, dtype=np.uint8)

        # Get the number of bytes
        num_bytes = len(bytes_data)

        # Convert each byte to a signed char representation
        signed_chars = [
            (int.from_bytes(byte, byteorder="little", signed=True))
            for byte in bytes_data
        ]

        # Split the array into chunks of 32 bytes, formatting each byte as an integer
        # and joining with ', ' within each chunk and '\n' between chunks
        formatted_data = ",\n".join(
            [
                ", ".join([str(b) for b in chunk])
                for chunk in np.array_split(signed_chars, len(signed_chars) / 32 + 1)
            ]
        )

    return formatted_data, num_bytes


if __name__ == "__main__":
    main()
