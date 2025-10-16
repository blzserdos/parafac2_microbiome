@echo off

:: Set variables
set ENV_NAME=pf2_micro
set YAML_FILE=requirements.yml

:: Initialize conda
:: Adjust path to your Anaconda/Miniconda install if different
CALL %USERPROFILE%\miniconda3\Scripts\activate.bat

:: Check if environment exists
conda env list | findstr /R "^%ENV_NAME% " >nul
if %ERRORLEVEL%==0 (
    echo Removing existing environment: %ENV_NAME%
    conda remove --name %ENV_NAME% --all -y
)

:: Create environment
echo Creating environment: %ENV_NAME%
conda env create -n %ENV_NAME% -f %YAML_FILE% python=3.12

:: Install additional package using conda run
echo Installing additional packages...
conda run -n %ENV_NAME% pip install --no-cache-dir matcouply-0.1.6.tar.gz

:: Activate environment (note: only persists inside this .bat run)
CALL conda activate %ENV_NAME%
:: jupyter notebook 2_reproduce_results.ipynb