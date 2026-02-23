# srsdk-model-target
Builds tooling to take a LiteRT model and target the SRSDK for the SR1xx and SRW1xxx series parts

## Links to the Vela compiler
https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela

Compiler options:
https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/blob/4.3.0/OPTIONS.md?ref_type=tags#configuration-file

## Get the initial REPO

Get the repo and requirements

## Development flow

This builds the block with the local package and supports development + testing

```bash
# Clone the repo
git clone git@github.com:syna-astra-mcu-dev/sr-model-compiler.git
cd sr-model-compiler

# Create Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install toolkit with dev tools
pip install -e .[dev]

# Run the full test sweep
make all

# format Python files
make format
```

### Running the command line compiler

```bash
usage: sr_model_compiler [-h] -m MODEL_FILE [--memory-mode {memory_sr100,memory_sl2600,memory_syn5312}]
                         [--system-config {.., }]
                         [--accel-config {..}] [--vmem-size-limit VMEM_SIZE_LIMIT] [--lpmem-size-limit LPMEM_SIZE_LIMIT]
                         [-o OUTPUT_DIR] [--model-namespace MODEL_NAMESPACE] [-n MODEL_FILE_OUT] [-s {model,inout} [{model,inout} ...]] [-i INPUT [INPUT ...]]
                         [-c {vela,synai,none}] [--arena-cache-size ARENA_CACHE_SIZE] [-v] [--verbose-cycle-estimate] [-p {Performance,Size}]

Wrapper script to compile a TFLite model onto SR devices.

options:
  -h, --help            show this help message and exit
  -m MODEL_FILE, --model-file MODEL_FILE
                        Path to TFLite model file
  --memory-mode {memory_sr100,memory_sl2600,memory_syn5312}
                        Sets memory architecture
  --system-config {sr100_npu_400MHz_all_vmem,sr100_npu_400MHz_tensor_vmem_weights_lpmem,sr100_npu_400MHz_tensor_vmem_weights_flash66MHz,sr100_npu_400MHz_tensor_vmem_weights_flash100MHz,sl2600_npu_1GHz_all_vmem,sl2600_npu_1GHz_tensor_vmem_weights_lpmem,sl2600_npu_1GHz_tensor_vmem_weights_flash66MHz,sl2600_npu_1GHz_tensor_vmem_weights_flash100MHz,syn5312_npu_400MHz_all_vmem,syn5312_npu_200MHz_tensor_vmem_weights_flash100MHz,syn5312_npu_200MHz_tensor_vmem_weights_flash48MHz}
                        Sets system config selection
  --accel-config {ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512, ethos-u85-256, ethos-u85-512}
                        Sets NPU size and version
  --vmem-size-limit VMEM_SIZE_LIMIT
                        Sets limit for vmem
  --lpmem-size-limit LPMEM_SIZE_LIMIT
                        Sets limit for lpmem (operates at 1/4 speed of vmem)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to output generated files
  --model-namespace MODEL_NAMESPACE
                        Sets the model namespace
  -n MODEL_FILE_OUT, --model-file-out MODEL_FILE_OUT
                        Name of the output cc file for the model
  -s {model,inout} [{model,inout} ...], --script {model,inout} [{model,inout} ...]
                        Choose specific scripts to run, if not provided then run all scripts
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        List of input npy/bin files
  -c {vela,synai,none}, --compiler {vela,synai,none}
                        Choose target compiler
  --arena-cache-size ARENA_CACHE_SIZE
                        Sets the model arena cache size in bytes
  -v, --verbose-all     Turns on verbose all for the compiler
  --verbose-cycle-estimate
                        Turns on verbose cycle estimation
  -p {Performance,Size}, --optimize {Performance,Size}
                        Choose optimization Type
```

TBD --accel-config
choices=[
            "ethos-u55-128",
            "ethos-u55-256",
            "ethos-u65-256",
            "ethos-u65-512",
            "ethos-u85-256",
            "ethos-u85-512",
        ],



### Running the command line optimizer

```bash
usage: sr100_model_optimizer [-h] -m MODEL_FILE [--vmem-size-limit VMEM_SIZE_LIMIT] [--lpmem-size-limit LPMEM_SIZE_LIMIT]
                             [-p {Performance,Size}]

Optimize memory location for a TFLite model for an SR100 devices.

options:
  -h, --help            show this help message and exit
  -m MODEL_FILE, --model-file MODEL_FILE
                        Path to TFLite model file
  --vmem-size-limit VMEM_SIZE_LIMIT
                        Set vmem size limit
  --lpmem-size-limit LPMEM_SIZE_LIMIT
                        Set lpmem size limit
  -p {Performance,Size}, --optimize {Performance,Size}
                        Choose optimization Type
```


### Testing Pipeline

Before you commit changes, make sure the test suite works

```bash
# Run format, lint and pytest tests
make all

# Format the Python
make format
```
