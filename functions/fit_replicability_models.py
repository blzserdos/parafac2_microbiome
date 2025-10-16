import sys
import os

sys.path.append("../")

from aux_funcs import *
import scipy.io as sio
from tensorly.decomposition import parafac, constrained_parafac
from tlviz.factor_tools import *
from sklearn.model_selection import RepeatedStratifiedKFold
import logging
import pickle
import time
from copy import deepcopy
from multiprocessing import Pool
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd

if __name__ == "__main__":

    dataset = sys.argv[1]
    method = sys.argv[2]
    rank = sys.argv[3]

    path = f"{dataset}/{method}/{rank}"
    if not os.path.exists(f"analysis_results/replicability/{path}"):
        os.makedirs(f"analysis_results/replicability/{path}", exist_ok=True)

    if dataset == "COPSAC2010":
        with open('data/COPSAC2010_data.pkl', 'rb') as f:
            Data = pickle.load(f)
        tensor, sub_id, tax_id = filter_data(Data, n_time=0, f_threshold=0.1)
        Metadata = pd.read_csv('data/COPSAC2010_metadata.csv', index_col=[0])
        Metadata = Metadata.loc[sub_id]
        strat_var = "Delivery mode"
        nfolds = 10
        gl_penalty = None

    elif dataset == "FARMM":
        Data = np.load("data/FARMM_data.npy")
        Data = np.moveaxis(Data, -1, -2).T # subjects by time by taxa
        tensor = np.delete(Data, 0, axis=1)
        Metadata = pd.read_csv('data/FARMM_metadata.csv', index_col=[0])
        Metadata = Metadata.groupby('SubjectID').agg({
            'study_day': 'first',
            'study_group': 'first'
        })
        strat_var = "study_group"
        nfolds = 5 
        M = 2 * np.eye(tensor.shape[1]) - np.eye(tensor.shape[1], k=1) - np.eye(tensor.shape[1], k=-1)
        M[0, 0] = 1
        M[-1, -1] = 1
        M = tl.tensor(M)
        gl_penalty = 1e-3*M # Graph Laplacian penalty


    rskf = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=10, random_state=42)
    rskf.get_n_splits(Metadata.index, Metadata[strat_var])
    
    args = []
    for split_no, (id_keep, id_drop) in enumerate(rskf.split(Metadata.index, Metadata[strat_var])):
        dropped_ids = Metadata.iloc[id_drop].index.to_list()
        split_metadata = Metadata.iloc[id_keep]
        split_tensor = tensor[id_keep,:,:]
        print(split_tensor.shape)

        split_tensor = clr(split_tensor)

        mask = np.isfinite(split_tensor)
     
        split_tensor = norm_tensor(split_tensor)

        if method == "cp":
            split_tensor = np.nan_to_num(split_tensor) # once masked, inf values must be set to numeric 
        
        ninits=40
        random_states = []
        for i in range(ninits):
            random_states.append(np.random.randint(0,1000000))

        l2_reg = 1e-3
        
        for init_no in random_states:

            args.append({
                "dataset_name": dataset,
                "method": method,
                "rank": rank,
                "tensor": deepcopy(split_tensor),
                "split_no": split_no,
                "init_no": init_no,
                "mask": deepcopy(mask),
                "l2_reg": l2_reg,
                "gl_penalty": gl_penalty,
                "dropped_ids": dropped_ids
            })

    pool = Pool(4) # number of parallel processes, adjust to your machine
    pool.map(fit_inits_, args)

    pool.close()
    pool.join()
