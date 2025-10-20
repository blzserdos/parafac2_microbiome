#!/bin/bash
ENV_NAME="pf2_micro"
YAML_FILE="requirements.yml"

# Delete conda environment if it exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Removing existing environment: $ENV_NAME"
    conda remove --name $ENV_NAME --all -y
fi

# Create environment from requirements.yml
echo "Creating environment: $ENV_NAME"
conda env create -n $ENV_NAME -f $YAML_FILE python=3.12.8

conda activate $ENV_NAME

echo "Installing additional packages..."
conda run -n $ENV_NAME pip install --no-cache-dir matcouply-0.1.6.tar.gz

conda install jupyter

jupyter notebook 2_reproduce_results.ipynb