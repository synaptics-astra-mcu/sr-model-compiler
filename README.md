# SR-MODEL-COMPILER
Python toolchain to build AI models for Synaptics SR100 and SRW1500 family of parts

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
git clone git@github.com:synaptics-astra-mcu/sr-model-compiler.git
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
# Grab latest command-line arguments
usage: sr_model_compiler [-h] -m MODEL_FILE  [...]

# Get help on all command line arguments
sr_model_compiler -h

# Get the available system-config and memory-mode settings
sr_model_compiler --get-modes
```

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
