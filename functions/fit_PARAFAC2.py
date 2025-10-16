import sys
import os
sys.path.append("../")

import pandas as pd
import scipy.io as sio
import logging
import pickle
import time
from aux_funcs import *
from tensorly.decomposition import parafac, constrained_parafac
import matcouply
from matcouply.decomposition import cmf_aoadmm
from matcouply.penalties import NonNegativity, Parafac2, GeneralizedL2Penalty
from tlviz.factor_tools import *
from copy import deepcopy
from multiprocessing import Pool

if __name__ == "__main__":

    dataset = sys.argv[1]
    method = sys.argv[2]
    rank = sys.argv[3]
    init = sys.argv[4]

    path = f"{dataset}/{method}/{rank}"
    if not os.path.exists(f"analysis_results/models/{path}/inits/"):
        os.makedirs(f"analysis_results/models/{path}/inits/", exist_ok=True)

    if not os.path.exists(f"analysis_results/models/logs/{path}/"):
        os.makedirs(f"analysis_results/models/logs/{path}/", exist_ok=True)

    log_file = f"analysis_results/models/logs/{path}/log.log"

    # Delete the log file if it already exists
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=f"analysis_results/models/logs/{path}/log.log",
        level=logging.INFO,
    )

    if dataset == "COPSAC2010":
        with open('data/COPSAC2010_data.pkl', 'rb') as f:
            Data = pickle.load(f)
        if "alt_1" in rank:
            tensor, sub_id, tax_id = filter_data(Data, n_time=0, f_threshold=0.2) # for alternative model 1 / replicability of subject specific factors
        elif "alt_2" in rank:
            tensor, sub_id, tax_id = filter_data(Data, n_time=0, f_threshold=0.3) # for alternative model 2 / replicability of subject specific factors
        else:
            tensor, sub_id, tax_id = filter_data(Data, n_time=0, f_threshold=0.1) # reference
        gl_penalty = None

    elif dataset == "FARMM":
        Data = np.load("data/FARMM_data.npy") 
        Data = np.moveaxis(Data, -1, -2).T
        tensor = np.delete(Data, 0, axis=1)
        M = 2 * np.eye(tensor.shape[1]) - np.eye(tensor.shape[1], k=1) - np.eye(tensor.shape[1], k=-1)
        M[0, 0] = 1
        M[-1, -1] = 1
        M = tl.tensor(M)
        gl_penalty = 1e-3*M # Graph Laplacian penalty

    # clr transform
    tensor = clr(tensor)
    mask = np.isfinite(tensor)

    # scale to norm 1
    tensor = norm_tensor(tensor)

    print(tensor.shape)
    if init == "paper_inits": # to exactly replicate models including saved initialization
        with open(f"analysis_results/models/{path}/inits.txt", 'r') as file: # to exactly replicate models including saved initialization
            random_states = [int(line.strip()) for line in file]
        ninit = len(random_states)
    else:
        random_states = []
        ninit = 60
        for i in range(ninit): # random initializations
            random_states.append(np.random.randint(0,1000000))

    logging.info("Input read successfully.")

    args = []

    l2_reg = 1e-3

    for init_no in random_states:
       
        args.append({
            "dataset_name": dataset,
            "method": method,
            "rank": rank,
            "tensor": deepcopy(tensor),
            "init_no": init_no,
            "mask": deepcopy(mask),
            "l2_reg": l2_reg,
            "gl_penalty": gl_penalty
        })

    pool = Pool(4) # number of parallel processes, adjust to your machine
    pool.map(fit_PARAFAC2_, args)

    pool.close()
    pool.join()
