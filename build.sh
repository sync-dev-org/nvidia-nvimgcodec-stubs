#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=".venv"

CUDA_VERSIONS=(
    "cu11"
    "cu12"
)

TARGET_VERSIONS=(
    "0.5.0.13"
)
POST_VERSION="5"

PY_VERSION="3.12"



# if not uv installed, install it
if ! command -v uv &> /dev/null; then
    echo "uv is not installed, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed"
fi


for CUDA_VERSION in "${CUDA_VERSIONS[@]}" ; do
    PROJECT_NAME="nvidia-nvimgcodec-${CUDA_VERSION}-stubs"
    TARGET_NAME="nvidia-nvimgcodec-${CUDA_VERSION}[all]"
    echo "Building for ${PROJECT_NAME} with target ${TARGET_NAME}"

    for TARGET_VERSION in "${TARGET_VERSIONS[@]}" ; do
        STUBS_VERSION="${TARGET_VERSION}.${POST_VERSION}"
        echo "Building Stubs ${STUBS_VERSION}"

        if [[ -d ${VENV_DIR} ]]; then
            echo "Virtual environment already exists, removing it..."
            rm -rf ${VENV_DIR}
        fi

        echo "Creating directory structure..."
        rm -rf src
        mkdir -p src

        cp ./template/pyproject.toml pyproject.toml
        sed -i "s/PROJECT_NAME/${PROJECT_NAME}/g" ./pyproject.toml
        sed -i "s/version = \"0.0.0.0\"/version = \"${STUBS_VERSION}\"/" ./pyproject.toml
        sed -i "s/TARGET_NAME==0.0.0.0/${TARGET_NAME}==${TARGET_VERSION}/" ./pyproject.toml

        cp ./template/README.md README.md
        sed -i "s/0.0.0.0/${TARGET_VERSION}/" ./README.md

        cp -r ./template/${PROJECT_NAME} src
        sed -i "s/__version__ = \"0.0.0.0\"/__version__ = \"${STUBS_VERSION}\"/" ./src/${PROJECT_NAME}/__init__.py

        echo "Creating virtual environment..."
        uv python pin ${PY_VERSION}
        uv sync --extra dev

        if [[ "$OSTYPE" == "msys"* ]]; then
            source ${VENV_DIR}/Scripts/activate
        else
            source ${VENV_DIR}/bin/activate
        fi

        python -m pybind11_stubgen nvidia --ignore-all-errors --output-dir src
        python -m pybind11_stubgen nvidia.nvcomp --ignore-all-errors --output-dir src
        python -m pybind11_stubgen nvidia.nvimgcodec --ignore-all-errors --output-dir src
        python -m pybind11_stubgen nvidia.nvjpeg --ignore-all-errors --output-dir src
        python -m pybind11_stubgen nvidia.nvjpeg2k --ignore-all-errors --output-dir src
        python -m pybind11_stubgen nvidia.nvtiff --ignore-all-errors --output-dir src
        uv build

        deactivate

        #twine upload --repository testpypi dist/nvidia_nvimgcodec_${CUDA_VERSION}_stubs-${STUBS_VERSION}-py3-none-any.whl
        twine upload --repository pypi dist/nvidia_nvimgcodec_${CUDA_VERSION}_stubs-${STUBS_VERSION}-py3-none-any.whl

    done

done
