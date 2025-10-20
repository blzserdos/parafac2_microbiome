#!/bin/bash
# 1. Run replicability analysis;

# COPSAC2010

# CP
# python functions/fit_replicability_models.py COPSAC2010 cp R1
# python functions/fit_replicability_models.py COPSAC2010 cp R2
# python functions/fit_replicability_models.py COPSAC2010 cp R3
# python functions/fit_replicability_models.py COPSAC2010 cp R4
# python functions/fit_replicability_models.py COPSAC2010 cp R5
# python functions/fit_replicability_models.py COPSAC2010 cp R6
# python functions/fit_replicability_models.py COPSAC2010 cp R7
# python functions/fit_replicability_models.py COPSAC2010 cp R8
# python functions/fit_replicability_models.py COPSAC2010 cp R9
# python functions/fit_replicability_models.py COPSAC2010 cp R10

# PARAFAC2
# python functions/fit_replicability_models.py COPSAC2010 parafac2 R1
# python functions/fit_replicability_models.py COPSAC2010 parafac2 R2
# python functions/fit_replicability_models.py COPSAC2010 parafac2 R3
# python functions/fit_replicability_models.py COPSAC2010 parafac2 R4
# python functions/fit_replicability_models.py COPSAC2010 parafac2 R5

# FARMM

# CP
# python functions/fit_replicability_models.py FARMM cp R1
# python functions/fit_replicability_models.py FARMM cp R2
# python functions/fit_replicability_models.py FARMM cp R3
# python functions/fit_replicability_models.py FARMM cp R4
# python functions/fit_replicability_models.py FARMM cp R5
# python functions/fit_replicability_models.py FARMM cp R6
# python functions/fit_replicability_models.py FARMM cp R7
# python functions/fit_replicability_models.py FARMM cp R8

# PARAFAC2
# python functions/fit_replicability_models.py FARMM parafac2 R1
# python functions/fit_replicability_models.py FARMM parafac2 R2
# python functions/fit_replicability_models.py FARMM parafac2 R3
# python functions/fit_replicability_models.py FARMM parafac2 R4
# python functions/fit_replicability_models.py FARMM parafac2 R5

# 2. Fit models to calculate fit and save final model for plotting

# COPSAC2010

# CP
# python functions/fit_CP.py COPSAC2010 cp R1 paper_inits
# python functions/collect_results.py COPSAC2010 cp R1
# python functions/fit_CP.py COPSAC2010 cp R2 paper_inits
# python functions/collect_results.py COPSAC2010 cp R2
# python functions/fit_CP.py COPSAC2010 cp R3 paper_inits
# python functions/collect_results.py COPSAC2010 cp R3
# python functions/fit_CP.py COPSAC2010 cp R4 paper_inits
# python functions/collect_results.py COPSAC2010 cp R4
# python functions/fit_CP.py COPSAC2010 cp R5 paper_inits
# python functions/collect_results.py COPSAC2010 cp R5
# python functions/fit_CP.py COPSAC2010 cp R6 paper_inits
# python functions/collect_results.py COPSAC2010 cp R6
# python functions/fit_CP.py COPSAC2010 cp R7 paper_inits
# python functions/collect_results.py COPSAC2010 cp R7
# python functions/fit_CP.py COPSAC2010 cp R8 paper_inits
# python functions/collect_results.py COPSAC2010 cp R8
# python functions/fit_CP.py COPSAC2010 cp R9 paper_inits
# python functions/collect_results.py COPSAC2010 cp R9
# python functions/fit_CP.py COPSAC2010 cp R10 paper_inits
# python functions/collect_results.py COPSAC2010 cp R10

# PARAFAC2
# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R1 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R1
# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R2 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R2
# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R3 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R3
# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R4 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R4
# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R5 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R5

# FARMM

# CP
# python functions/fit_CP.py FARMM cp R1 paper_inits
# python functions/collect_results.py FARMM cp R1
# python functions/fit_CP.py FARMM cp R2 paper_inits
# python functions/collect_results.py FARMM cp R2
# python functions/fit_CP.py FARMM cp R3 paper_inits
# python functions/collect_results.py FARMM cp R3
# python functions/fit_CP.py FARMM cp R4 paper_inits
# python functions/collect_results.py FARMM cp R4
# python functions/fit_CP.py FARMM cp R5 paper_inits
# python functions/collect_results.py FARMM cp R5
# python functions/fit_CP.py FARMM cp R6 paper_inits
# python functions/collect_results.py FARMM cp R6
# python functions/fit_CP.py FARMM cp R7 paper_inits
# python functions/collect_results.py FARMM cp R7
# python functions/fit_CP.py FARMM cp R8 paper_inits
# python functions/collect_results.py FARMM cp R8

# PARAFAC2
# python functions/fit_PARAFAC2.py FARMM parafac2 R1 paper_inits
# python functions/collect_results.py FARMM parafac2 R1
# python functions/fit_PARAFAC2.py FARMM parafac2 R2 paper_inits
# python functions/collect_results.py FARMM parafac2 R2
# python functions/fit_PARAFAC2.py FARMM parafac2 R3 paper_inits
# python functions/collect_results.py FARMM parafac2 R3
# python functions/fit_PARAFAC2.py FARMM parafac2 R4 paper_inits
# python functions/collect_results.py FARMM parafac2 R4
# python functions/fit_PARAFAC2.py FARMM parafac2 R5 paper_inits
# python functions/collect_results.py FARMM parafac2 R5


# 3. Fit alternative models to evaluate replicability of subject specific factors

# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R5_alt_1 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R5_alt_1

# python functions/fit_PARAFAC2.py COPSAC2010 parafac2 R5_alt_2 paper_inits
# python functions/collect_results.py COPSAC2010 parafac2 R5_alt_2