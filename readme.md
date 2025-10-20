## Code and data accompanying the manuscript: Extracting host-specific developmental signatures from longitudinal microbiome data
---------------
This repository contains the data and code to reproduce the findings and figures presented in our manuscript. This work introduces a new method to analyze longitudinal microbiome data to identify microbial patterns linked to host development.

### Data

The processed data used in the analysis is located in the `/data` directory. The raw sequencing data is available at the NCBI Sequence Read Archive under accession number [Your Accession Number].

### Requirements

  * [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda/Anaconda/Mamba)
  * Python 3.12
  * OS: Linux, MacOS, or Windows

### Reproducing the Analysis

This analysis can be fully reproduced in two main steps.

**Step 1: Setup the Environment** 🖥️

First, set up the Conda environment, which includes all necessary packages. This should take about 5-10 minutes.

  * For Linux/MacOS:
    ```bash
    ./setup.sh
    ```
  * For Windows:
    ```bash
    ./setup.bat
    ```

This script creates a Conda environment named `pf2_micro` from the `environment.yml` file and launches the Jupyter Notebook `2_reproduce_results.ipynb`.

**Step 2: Run the Analysis Notebook** 🔬

1.  Once Jupyter opens, navigate to and run the `2_reproduce_results.ipynb` notebook.
2.  Running all cells in this notebook will reproduce the results and save all manuscript figures (Figures 1-5 and Supplementary Figures S1-S11) to the `/figures` directory. The notebook uses pre-computed model factors for speed.

**(Optional) Re-fitting the Tensor Models**

If you wish to re-fit the CP and PARAFAC2 models from scratch, follow these steps. **Warning**: This is computationally intensive and may take several hours depending on the analysis.

1.  Open `1_fit_model.sh` (Linux/MacOS) or `1_fit_model.bat` (Windows) and uncomment the lines corresponding to the models you want to re-fit
2.  Run the model fitting script:
      * For Linux/MacOS:
        ```bash
        ./1_fit_model.sh
        ```
      * For Windows:
        ```bash
        ./1_fit_model.bat
        ```
    This will overwrite the pre-saved factors in the `/analysis_results` directory.
3.  In the `2_reproduce_results.ipynb` notebook, uncomment the cells under *Model selection* and re-run the notebook to generate figures using the re-fitted models.

## Directory Structure

```
.
├── data/               # Processed data files
├── analysis_results/   # Output directory for model results
│   └── figures/        # Output directory for generated figures
│   └── models/         # Output directory for estimated model factors and diagnostics
│   └── replicability/  # Output directory for replicability analysis results
├── environment.yml     # Conda environment specification
├── 1_fit_model.sh      # (Optional) Script to re-fit models
└── 2_reproduce_results.ipynb # Main notebook for analysis and plotting
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or issues, please contact [Balazs Erdos] at [erdos.blz@gmail.com].