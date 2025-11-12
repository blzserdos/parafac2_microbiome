#!/bin/bash
ENV_NAME="pf2_micro"
YAML_FILE="requirements.yml"

# Delete conda environment if it exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment: $ENV_NAME"
    conda remove --name $ENV_NAME --all -y
fi

# 1. Create environment from YAML file
echo "Creating environment: $ENV_NAME"
conda env create -f $YAML_FILE -f $YAML_FILE python=3.12.8

# 2. Install additional pip package
echo "Installing additional pip package..."
conda run -n $ENV_NAME pip install --no-cache-dir matcouply-0.1.6.tar.gz

# 3. Install jupyter AND ipykernel INSIDE the environment
echo "Installing jupyter and ipykernel in $ENV_NAME..."
conda run -n $ENV_NAME conda install -y jupyter ipykernel

# 4. Register the environment as a Jupyter kernel
echo "Registering $ENV_NAME as a jupyter kernel..."
conda run -n $ENV_NAME python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# 5. Launch jupyter notebook
echo "Launching Jupyter Notebook..."
conda run -n $ENV_NAME jupyter notebook 2_reproduce_results.ipynb