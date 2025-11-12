@echo off
:: This is a Windows Batch script to create the conda environment.

set "ENV_NAME=pf2_micro"
set "YAML_FILE=requirements.yml"

:: 1. Delete conda environment if it exists
echo "Checking for existing environment: %ENV_NAME%"
conda info --envs | findstr /B /C:"%ENV_NAME% " > NUL
if %ERRORLEVEL% == 0 (
    echo "Removing existing environment: %ENV_NAME%"
    conda remove --name %ENV_NAME% --all -y
) else (
    echo "Environment %ENV_NAME% not found, proceeding."
)

:: 2. Create environment from YAML file
echo "Creating environment: %ENV_NAME%"
conda env create -f %YAML_FILE%

:: 3. Install additional pip package
echo "Installing additional pip package..."
conda run -n %ENV_NAME% pip install --no-cache-dir matcouply-0.1.6.tar.gz

:: 4. Install jupyter AND ipykernel
echo "Installing jupyter and ipykernel in %ENV_NAME%..."
conda run -n %ENV_NAME% conda install -y jupyter ipykernel

:: 5. Register the environment as a Jupyter kernel
echo "Registering %ENV_NAME% as a jupyter kernel..."
conda run -n %ENV_NAME% python -m ipykernel install --user --name "%ENV_NAME%" --display-name "Python (%ENV_NAME%)"

:: 6. Launch jupyter notebook
echo "Launching Jupyter Notebook..."
conda run -n %ENV_NAME% jupyter notebook 2_reproduce_results.ipynb

echo "Script finished."