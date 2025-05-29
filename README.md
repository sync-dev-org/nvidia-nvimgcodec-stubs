# nvidia-nvimgcodec-stubs

Type stubs for NVIDIA nvimgcodec Python API.

## Description

This project provides type stubs for the NVIDIA nvimgcodec Python API. It enables static type checking and IDE autocompletion for nvimgcodec in Python projects.

## Features

- Support for both CUDA 11 and CUDA 12
- Includes type stubs for the following NVIDIA libraries:
  - `nvidia.nvcomp` - Data compression library
  - `nvidia.nvimgcodec` - Image encoding/decoding library
  - `nvidia.nvjpeg` - JPEG processing library
  - `nvidia.nvjpeg2k` - JPEG 2000 processing library
  - `nvidia.nvtiff` - TIFF processing library

## Installation

### CUDA 12 version
```bash
pip install nvidia-nvimgcodec-cu12-stubs
```

### CUDA 11 version
```bash
pip install nvidia-nvimgcodec-cu11-stubs
```

## Usage

After installation, your IDE and type checkers (like mypy, pyright, etc.) will automatically use these stubs when you import nvimgcodec:

```python
import nvidia.nvimgcodec as nvimgcodec

# Your IDE will now provide autocompletion and type checking for nvimgcodec
decoder = nvimgcodec.Decoder()
```

## Requirements

- Python >= 3.10
- nvidia-nvimgcodec-cu12[all] == 0.5.0.13 (for CUDA 12 version)
- nvidia-nvimgcodec-cu11[all] == 0.5.0.13 (for CUDA 11 version)

## Development

### Build Environment Setup

This project uses `uv` for building:

```bash
# uv will be automatically installed if not present
./build.sh
```

### Build Process

The `build.sh` script automatically performs the following:

1. Generates stubs for both CUDA 11 and CUDA 12 versions
2. Creates project configuration from template files
3. Auto-generates type stubs using `pybind11-stubgen`
4. Builds the packages

### Manual Stub Generation

```bash
# Create and activate virtual environment
uv python pin 3.12
uv sync --extra dev
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Generate stubs
python -m pybind11_stubgen nvidia --ignore-all-errors --output-dir src
python -m pybind11_stubgen nvidia.nvcomp --ignore-all-errors --output-dir src
python -m pybind11_stubgen nvidia.nvimgcodec --ignore-all-errors --output-dir src
python -m pybind11_stubgen nvidia.nvjpeg --ignore-all-errors --output-dir src
python -m pybind11_stubgen nvidia.nvjpeg2k --ignore-all-errors --output-dir src
python -m pybind11_stubgen nvidia.nvtiff --ignore-all-errors --output-dir src

# Build package
uv build
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Links

- [NVIDIA nvimgcodec](https://github.com/NVIDIA/nvImageCodec)
- [GitHub Repository](https://github.com/sync-dev-org/nvidia-nvimgcodec-stubs)