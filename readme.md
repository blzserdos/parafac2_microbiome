[![DOI](https://zenodo.org/badge/1077451759.svg)](https://doi.org/10.5281/zenodo.17674126)

## Code and data accompanying the manuscript: Extracting host-specific developmental signatures from longitudinal microbiome data
---------------
This repository contains data and code to reproduce the findings and figures presented in the manuscript.

### Data Availability
Shotgun metagenomic sequence data from the FARMM dataset [Tanes et al., 2021](https://doi.org/10.1016/j.chom.2020.12.012) were deposited under BioProject with accession code PRJNA675301. Processed data were obtained from [Ma and Li, 2023](https://doi.org/10.1214/22-AOAS1661) and is located in the `/data` directory. Individual-level clinical data from the COPSAC<sub>2010</sub> cohort are not publicly available to protect participant privacy, in accordance with the Danish Data Protection Act and European Regulation 2016/679 of the European Parliament and of the Council (GDPR) that prohibit distribution even in pseudo-anonymized form. Data can be made available under a joint research collaboration by contacting COPSAC’s data protection officer (administration@dbac.dk). 

### Requirements

  * [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda/Anaconda/Mamba)
  * Python 3.12
  * OS: Linux, MacOS, or Windows (Windows users may need to install dependencies manually)

### Reproducing the Analysis

The analysis can be reproduced in two main steps.

**Step 1: Setup the environment** 🖥️

First, set up the Conda environment, which includes all necessary packages. This should take about 5-10 minutes.

  * For Linux/MacOS:
    ```bash
    ./setup.sh
    ```
  * For Windows (this is experimental and may require manual installation of some dependencies):
    ```bash
    ./setup.bat
    ```

This script creates a Conda environment named `pf2_micro` from the `environment.yml` file and launches the Jupyter Notebook `2_reproduce_results.ipynb`.

**Step 2: Run the analysis notebook** 🔬

1.  Once Jupyter opens, navigate to and run the `2_reproduce_results.ipynb` notebook.
2.  Running all cells in this notebook will reproduce the results and save manuscript figures (Figures 1-5 and Supplementary Figures S1-S11) to the `/figures` directory. The notebook uses pre-computed model factors for speed.

**(Optional) Re-fitting the tensor decomposition models**

If you wish to re-fit the CP and PARAFAC2 models from scratch, follow these steps. **Warning**: This is computationally intensive and may take several hours depending on the analysis.

1.  Open `1_fit_model.sh` (Linux/MacOS) or `1_fit_model.bat` (Windows) and uncomment the lines corresponding to the models you want to re-fit. For example, if you wish to re-run a 3-component CP model on the FARMM data you should uncomment the following lines:
    ``` 
    python functions/fit_CP.py FARMM cp R3 paper_inits
    python functions/collect_results.py FARMM cp R3
    ```
The first line fits the model, while the second collects all factors computed, discards unfeasible and degenerate solutions and chooses the best run according to lowest reconstruction error, saving it in `analysis_results/models/FARMM/cp/R3/best_run.pkl`. The optional `paper_inits` argument fixes the model initializations to the ones used in the manuscript to ensure reproducibility, otherwise random initializations are used.

2.  Run the model fitting script:
      * For Linux/MacOS:
        ```bash
        ./1_fit_model.sh
        ```
      * For Windows:
        ```bash
        ./1_fit_model.bat
        ```
    This will overwrite the pre-saved factors in the `analysis_results/models/FARMM/cp/R3/best_run.pkl` directory.
    
3.  In the `2_reproduce_results.ipynb` notebook, uncomment the cells under *Model selection* in order to recompute the fit and FMS for the model selection results, and re-run the notebook to generate figures using the re-fitted models.

## Directory Structure

```
.
├── data/               # Processed data files
├── analysis_results/   # Output directory for model results
│   └── figures/        # Output directory for generated figures
│   └── models/         # Output directory for estimated model factors and diagnostics
│   └── replicability/  # Output directory for replicability analysis results
├── functions/          # Python functions for model fitting, analysis, plotting
├── environment.yml     # Conda environment specification
├── 1_fit_model.sh      # (Optional) Script to re-fit models
├── 2_reproduce_results.ipynb # Main notebook for analysis and plotting
├── 3_reproduce_results_rev.ipynb # Additional analysis at revision and plotting
├── matcouply-0.1.6.tar.gz # MatCouply library including missing data handling
├── setup.sh            # Setup script for Linux/MacOS
├── setup.bat           # Setup script for Windows (experimental, eventual manual installation of dependencies may be needed)
└── example_script.ipynb  # Example script with basic usage of PARAFAC2 model on microbiome data
```

## Citation info

```
@article {ErdosExtracting2026,
	author = {Erd{\H o}s, Bal{\'a}zs and Chatzis, Christos and Thorsen, Jonathan and Stokholm, Jakob and Smilde, Age K. and Rasmussen, Morten A. and Acar, Evrim},
	title = {Extracting host-specific developmental signatures from longitudinal microbiome data},
	elocation-id = {2025.11.22.689760},
	year = {2026},
	doi = {10.1101/2025.11.22.689760},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/28/2025.11.22.689760},
	journal = {bioRxiv}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or issues, please contact [Balazs Erdos] at [erdos.blz@gmail.com].
