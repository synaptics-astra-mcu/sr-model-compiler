import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from mako.template import Template
from pathlib import Path
import platform


def generate_input_expected_data(
    tflite_path, output_folder, namespace, license_header, input_files=None
):
    # Load the model
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path,
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Generate input and output data for each input and output
    input_data_list = []
    output_data_list = []
    input_data_size_list = []
    output_data_size_list = []
    for i, input_detail in enumerate(input_details):
        input_shape = input_detail["shape"]
        input_shape_bytes = 1
        for dim in input_shape:
            input_shape_bytes *= dim
        input_data = None
        input_found = False
        input_type = input_detail["dtype"]

        if (input_files != None) and (i < len(input_files)):
            # print(f"input_files : {input_files[i]}")
            file_extension = os.path.splitext(input_files[i])[1].lower()
            # npy_name = f"{input_folder}\input_{i}.npy"
            # bin_name = f"{input_folder}\input_{i}.bin"
            if file_extension == ".npy":
                print(f"Trying to load input {i} from: {input_files[i]}")
                data = np.load(input_files[i])
                if data.size == input_shape_bytes:
                    input_data = data
                    input_found = True
            elif file_extension == ".bin":
                print(f"Trying to load input {i} from: {input_files[i]}")
                with open(input_files[i], "rb") as file:
                    # Read the binary data from the file into a variable
                    bin_data = file.read()
                    input_data = np.frombuffer(bin_data, dtype=np.int8).reshape(
                        input_shape
                    )
                    input_found = True

        if not input_found:
            print(f"User input not found, generating random input for input {i}")
            input_data = np.random.randint(-128, 127, size=input_shape).astype("int8")
        else:
            print(f"User input loaded for input {i}")

        input_data_str = ",\n".join(
            [
                ", ".join([str(b) for b in a])
                for a in np.array_split(
                    input_data.flatten(), input_data.nbytes / 32 + 1
                )
            ]
        )
        interpreter.set_tensor(input_detail["index"], input_data)
        input_data_list.append(input_data_str)
        input_data_size_list.append(input_shape_bytes)

    interpreter.invoke()

    for i, output_detail in enumerate(output_details):
        output_data = interpreter.get_tensor(output_detail["index"])
        output_data_str = ",\n".join(
            [
                ", ".join([str(b) for b in a])
                for a in np.array_split(
                    output_data.flatten(), output_data.nbytes / 32 + 1
                )
            ]
        )
        output_data_list.append(output_data_str)
        output_data_size_list.append(output_data.nbytes)

        # Write output_data to binary file
        bin_filename = f"{output_folder}/output_{i}.bin"
        with open(bin_filename, "wb") as bin_file:
            bin_file.write(output_data.tobytes())

        # Write output_data to NumPy file
        npy_filename = f"{output_folder}/output_{i}.npy"
        np.save(npy_filename, output_data)

    # Get the path to the directory containing this script
    script_dir = Path(__file__).parent

    # Construct the relative path to the template file
    template_path = script_dir / "templates" / "io_template.mako"

    # Generate the C++ code from the Mako template
    template = Template(filename=str(template_path))
    output = template.render(
        namespace=namespace,
        input_data_list=input_data_list,
        output_data_list=output_data_list,
        input_data_size_list=input_data_size_list,
        output_data_size_list=output_data_size_list,
    )
    output = output.replace("\n", " ")
    output = license_header + "\n" + output
    # Write the generated code to a file
    filename = f"{output_folder}/{namespace}_io.cc"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)

    if platform.system() == "Windows":
        print(
            f"++ Generated input and expected output of {os.path.basename(tflite_path)} to {os.path.abspath(output_folder)}\{namespace}_io.cc"
        )
    else:
        print(
            f"++ Generated input and expected output of {os.path.basename(tflite_path)} to {os.path.abspath(output_folder)}/{namespace}_io.cc"
        )


# Optionally, keep the command-line interface for standalone usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Run inference on a TensorFlow Lite model and generate input/output arrays in C++ format"
    )
    parser.add_argument("-t", "--tflite_path", help="Input tflite file", required=True)
    parser.add_argument("-o", "--output_folder", help="Output folder", required=True)
    parser.add_argument("-n", "--namespace", help="namespace name", required=True)
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Folder containing input npy files with input_x.npy format",
    )
    args = parser.parse_args()

    license_header = ""

    generate_input_expected_data(
        args.tflite_path, args.output_folder, args.namespace, license_header, args.input
    )
