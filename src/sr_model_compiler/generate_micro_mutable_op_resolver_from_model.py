import os
import re
from mako.template import Template
from tensorflow.lite.tools import visualize as visualize
from mako import template
from pathlib import Path
import platform


def generate_micro_mutable_ops_resolver_header(
    common_tflite_path,
    input_tflite_files,
    output_dir,
    namespace,
    license_header,
    verify_op_list_against_header=None,
):
    TEMPLATE_DIR = os.path.abspath("templates")

    def parse_string(word):
        """Converts a flatbuffer operator string to a format suitable for Micro
        Mutable Op Resolver. Example: CONV_2D --> AddConv2D."""

        # Edge case for AddDetectionPostprocess().
        # The custom code is TFLite_Detection_PostProcess.
        word = word.replace("TFLite", "")

        word_split = re.split("_|-", word)
        formated_op_string = ""
        for part in word_split:
            if len(part) > 1:
                if part[0].isalpha():
                    formated_op_string += part[0].upper() + part[1:].lower()
                else:
                    formated_op_string += part.upper()
            else:
                formated_op_string += part.upper()
        return "Add" + formated_op_string

    def GetModelOperatorsAndActivation(model_path):
        """Extracts a set of operators from a tflite model."""

        custom_op_found = False
        operators_and_activations = set()

        print(f"Trying to open {model_path}")
        with open(model_path, "rb") as f:
            data_bytes = bytearray(f.read())

        data = visualize.CreateDictFromFlatbuffer(data_bytes)

        for op_code in data["operator_codes"]:
            if op_code["custom_code"] is None:
                op_code["builtin_code"] = max(
                    op_code["builtin_code"], op_code["deprecated_builtin_code"]
                )
            else:
                custom_op_found = True
                operators_and_activations.add(
                    visualize.NameListToString(op_code["custom_code"])
                )

        for op_code in data["operator_codes"]:
            # Custom operator already added.
            if (
                custom_op_found
                and visualize.BuiltinCodeToName(op_code["builtin_code"]) == "CUSTOM"
            ):
                continue

            operators_and_activations.add(
                visualize.BuiltinCodeToName(op_code["builtin_code"])
            )

        return operators_and_activations

    def GenerateMicroMutableOpsResolverHeaderFile(
        operators, name_of_model, output_dir, namespace
    ):
        """Generates Micro Mutable Op Resolver code based on a template."""

        number_of_ops = len(operators)
        outfile = "micro_mutable_op_resolver.hpp"

        # Get the path to the directory containing this script
        script_dir = Path(__file__).parent

        # Construct the relative path to the template file
        template_file_path = script_dir / "templates" / (outfile + ".mako")

        # Generate the resolver file with the template
        build_template = Template(filename=str(template_file_path))

        output_dir = Path(output_dir).resolve()
        if platform.system() == "Windows":
            output_path = str(output_dir) + "\\" + (namespace + "_" + outfile)
        else:
            output_path = str(output_dir) + "/" + (namespace + "_" + outfile)

        with open(output_path, "w") as file_obj:
            key_values_in_template = {
                "model": name_of_model,
                "number_of_ops": number_of_ops,
                "operators": operators,
                "namespace": namespace,
                "common_template_header": license_header,
            }
            file_obj.write(build_template.render(**key_values_in_template))

    def verify_op_list(op_list, header):
        """
        Verifies that all operations in op_list are supported by TFLM, as declared in the header file.

        Args:
            op_list (list): A list of operation names to verify.
            header (str): Path to the header file containing declarations of supported operations.

        Returns:
            bool: True if any operation in op_list is not supported, False otherwise.
        """
        # Read the header file and extract supported operations
        supported_op_list = []
        with open(header, "r") as f:
            for line in f:
                # Assuming the header file declares operations in the form "TfLiteStatus Add<OpName>(...);"
                match = re.search(r"TfLiteStatus Add(\w+)\(.*\);", line)
                if match:
                    supported_op = match.group(1)
                    supported_op_list.append(supported_op)

        # Check if all operations in op_list are in supported_op_list
        unsupported_ops = [op for op in op_list if op not in supported_op_list]
        if unsupported_ops:
            print(
                f"The following operations are not supported by TFLM: {', '.join(unsupported_ops)}"
            )
            return True  # Indicating verification failed due to unsupported operations

        return False  # All operations are supported

    model_names = []
    final_operator_list = []
    merged_operator_list = []

    for relative_model_path in input_tflite_files:
        full_model_path = f"{common_tflite_path}/{relative_model_path}"
        operators = GetModelOperatorsAndActivation(full_model_path)
        model_name = os.path.basename(full_model_path)
        model_names.append(model_name)

        parsed_operator_list = [parse_string(op) for op in sorted(operators)]
        merged_operator_list.extend(parsed_operator_list)

    final_operator_list = sorted(set(merged_operator_list))

    if verify_op_list_against_header:
        if verify_op_list(final_operator_list, verify_op_list_against_header):
            print("Verification failed.")
            return

    os.makedirs(output_dir, exist_ok=True)
    GenerateMicroMutableOpsResolverHeaderFile(
        final_operator_list, model_name, output_dir, namespace
    )


# Optionally, keep the command-line interface for standalone usage
if __name__ == "__main__":
    from absl import app
    from absl import flags

    FLAGS = flags.FLAGS
    flags.DEFINE_string("common_tflite_path", None, "Common path to tflite files.")
    flags.DEFINE_list("input_tflite_files", None, "List of input TFLite files.")
    flags.DEFINE_string("output_dir", None, "Directory to output generated files.")
    flags.DEFINE_string("namespace", None, "Namespace for the generated code.")
    flags.DEFINE_string("license_header", None, "License header")
    flags.DEFINE_string(
        "verify_op_list_against_header",
        None,
        "Header file to verify the operation list against.",
    )

    flags.mark_flag_as_required("common_tflite_path")
    flags.mark_flag_as_required("input_tflite_files")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("namespace")
    flags.mark_flag_as_required("license_header")

    def main(argv):
        print("generate_micro_mutable_ops_resolver_header")
        generate_micro_mutable_ops_resolver_header(
            FLAGS.common_tflite_path,
            FLAGS.input_tflite_files,
            FLAGS.output_dir,
            FLAGS.namespace,
            FLAGS.license_header,
            FLAGS.verify_op_list_against_header,
        )

    app.run(main)
