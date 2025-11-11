import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append("../")
import logging
import pickle
import time
import numpy as np
import pandas as pd
import tensorly as tl
import numpy.linalg as la
import re
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from statannotations.Annotator import Annotator
from copy import deepcopy
from scipy import stats
from tensorly.decomposition import parafac, constrained_parafac
import matcouply
from matcouply.penalties import NonNegativity, Parafac2, MatricesPenalty, GeneralizedL2Penalty, L1Penalty
from matcouply.decomposition import cmf_aoadmm, initialize_cmf
from tensorly.metrics import congruence_coefficient
from matcouply._utils import get_svd
from natsort import natsorted, natsort_keygen


from tlviz.factor_tools import factor_match_score, cosine_similarity, degeneracy_score

from tensorly.cp_tensor import CPTensor

def pattern1(x): # bell
    y = 0.5 * np.sin(x * np.pi * 2 - np.pi / 2) + 0.5
    return y

def pattern2(x,z): # subject specific sigmoid
    z_ = -0.073*z
    y = 1/(1+np.exp(-24*(1+z_)*x+5+0.11*z))
    return y

def pattern3(x): # sin
    y = 0.5 * np.sin(x * np.pi * 4 - np.pi / 2) + 0.5
    return y

def get_fms(gnd_factors, est_factors, skip_mode=None):

    (A, B_is, C) = gnd_factors
    (A2, B_is2, C2) = est_factors

    cp_tensor1 = (
        (np.array([1.0] * A.shape[1])),
        (A, np.vstack(np.array(deepcopy(B_is))), C),
    )
    cp_tensor2 = (
        (np.array([1.0] * A2.shape[1])),
        (A2, np.vstack(np.array(deepcopy(B_is2))), C2),
    )

    return factor_match_score(
            cp_tensor1,
            cp_tensor2,
            absolute_value=True,
            consider_weights=False,
            skip_mode=skip_mode)

def filter_data(Data, n_time=1, f_threshold=0.1):
    """
    Filters subjects based on missing time points and a presence threshold.

    Parameters:
    X: A numpy structured object with `X.data` as a 3D array (subjects × features × time points).
    ntime: Number of missing time points allowed.
    threshold: Proportion of individuals per time point in which the feature is present.

    Returns:
    Y: Filtered dataset.
    id: Indices of selected features.
    """

    tensor = Data['data']
    tensor = np.moveaxis(tensor, 1, 2)
    subject_ids = Data['ids']

    # Identify NaN values
    nans = np.isnan(tensor)

    # Compute mode along axis 2 (time points) to identify completely missing microbes per subject
    nan_time = np.apply_along_axis(lambda arr: np.bincount(arr).argmax(), axis=2, arr=nans.astype(int))

    # Select subjects where the number of missing time points is within the allowed limit
    subj_id = [k for k in range(tensor.shape[0]) if np.sum(nan_time[k, :]) <= n_time]

    # Filter X to keep only selected subjects
    X = tensor[subj_id, :, :]

    # Identify complete cases (features present at each time point)
    keep = ~np.isnan(X).any(axis=(-1))

    # Compute M (nonzero counts) and S (sum of values) across subjects for each feature and time point
    M = np.zeros((X.shape[2], X.shape[1]))
    S = np.zeros((X.shape[2], X.shape[1]))

    for k in range(X.shape[1]):  # Iterate over time points
        for j in range(X.shape[2]):  # Iterate over features
            # valid_indices = np.where(complete[:, j, k])[0]  # Indices of complete cases
            M[j, k] = np.count_nonzero(X[keep[:,k],k,j])  # Count of nonzero entries
            S[j, k] = np.sum(X[keep[:,k],k,j])  # Sum of valid values

    # Compute the threshold number of individuals per time point
    thres = f_threshold * X.shape[0]

    # Select feature indices where at least one time point has more than `thres` individuals
    tax_id = [i for i in range(M.shape[0]) if np.any(M[i, :] > thres)]
    subj_id = subject_ids[subj_id]
    
    return X[:, :, tax_id], subj_id, tax_id

def clr(X):
    Xclr = np.empty(X.shape)
    mask = np.isfinite(X) # if contains missing!
    X[mask] = X[mask] + 0.5 # add pseudocount
    for i in range(X.shape[1]):

        lmat = np.log(X[:,i,:], out=np.full_like(X[:,i,:], np.nan, dtype=np.double), where=~np.isnan(X[:,i,:]))
        gm = lmat.mean(axis=-1, keepdims=True)
        Xclr[:,i,:] = np.subtract(lmat, gm).squeeze()

    return Xclr

def norm_tensor(tensor):
    # if there are missing values the norm is computed only on the non-missing values

    normZ = 0
    for k in range(tensor.shape[0]):
        normZ += np.linalg.norm(tensor[k,~np.all(np.isnan(tensor[k,:,:]), axis=1),:], 'fro')**2

    normZ = np.sqrt(normZ)
    for k in range(tensor.shape[0]):
        tensor[k,~np.all(np.isnan(tensor[k,:,:]), axis=1),:] = tensor[k,~np.all(np.isnan(tensor[k,:,:]), axis=1),:] / normZ
        
    return tensor

def percentage_nan(arr):
  """Calculates the percentage of NaN values in a NumPy array.

  Args:
    arr: The NumPy array.

  Returns:
    The percentage of NaN values in the array.
  """
  total_elements = arr.size
  nan_count = np.isnan(arr).sum()
  if total_elements == 0:
        return 0.0
  return (nan_count / total_elements) * 100

def check_degenerate(factors, method, threshold=-0.85):
    """
    Check solution for degenerecy (wrapper for tlviz degeneracy score).
    """
    if method  in ("parafac2", "cmf"):

        A = factors[2]
        B = factors[1]
        D = factors[0]

        new_B = np.vstack(B)
        decomp = CPTensor((np.ones(A.shape[1]), (D, new_B, A)))

    elif method == "cp":

        decomp = factors

    if degeneracy_score(decomp) < threshold:
        return True
    else:
        return False

def get_model_fit(factors, data, mask):
    """
    Compute model fit PARAFAC2  -- DONE FOR WEIGHTS ABSORBED IN FACTORS
    """

    A = factors[2]
    B = factors[1]
    C = factors[0]

    fit = 0
    norm = 0

    for k in range(len(B)):
        norm += tl.norm(data[k,:,:][mask[k,:,:].sum(axis=1) > 0])**2
        recreated = B[k] @ np.diag(C[k,:]) @ A.T
        recreated = recreated[mask[k,:,:].sum(axis=1) > 0, :]
        fit += tl.norm(data[k,:,:][mask[k,:,:].sum(axis=1) > 0] - recreated)**2

    return 100*(1 - fit/norm)

def reconstructed_variance(tFac, tIn=None):
    """This function calculates the amount of variance captured (R2X) by the tensor method."""
    tMask = np.isfinite(tIn)
    vTop = np.sum(np.square(tl.cp_to_tensor(tFac) * tMask - np.nan_to_num(tIn)))
    vBottom = np.sum(np.square(np.nan_to_num(tIn)))
    return 1.0 - vTop / vBottom

def topn_subj(CB1, CB2, comp, topn):
    # find topn most different and most similar subjects in terms of scaled time loadings in component comp between two factorization results CB1 and CB2
    d = CB1 - CB2
    abs_row_means = np.mean(np.abs(d[:,:,comp]), axis=1)
    max_ixs = np.argpartition(abs_row_means,-topn)[-topn:] # top n largest
    min_ixs = np.argpartition(abs_row_means,topn)[:topn] # top n smallest
    
    return max_ixs, min_ixs, abs_row_means 

def filter_iterations(group):
    if (group['iterations'] >= 8000).all():
        return group.loc[group['final_rec_error'].idxmin()]
    else:
        filtered_group = group[group['iterations'] < 8000] # change 8000 to your max_iters!
        if filtered_group['final_rec_error'].min() < group['final_rec_error'].min():
            return filtered_group.loc[filtered_group['final_rec_error'].idxmin()]
        else:
            return group.loc[group['final_rec_error'].idxmin()]
    
def scale_factors(factors, method):

    if method in ("parafac2", "cmf"):
        # normalize factors C, B, A from PARAFAC2 model
        # by putting weights into C. 
        # Assumes that B varies according to levels of C.
        # due to matcouply convention, factors should be in order C,B,A
        (C,B,A) = factors

        As = np.empty(A.shape)
        Bs = deepcopy(B)
        Cs = np.empty(C.shape)

        R = A.shape[1]
        K = C.shape[0]
        for r in range(R):
            norm_Ar = tl.norm(A[:, r])
            As[:, r] = A[:, r] / norm_Ar
            Cs[:, r] = C[:, r] * norm_Ar

            for k in range(K):
                norm_Brk = tl.norm(B[k][:, r])
                Bs[k][:, r] = B[k][:, r] / norm_Brk
                Cs[k, r] = Cs[k, r] * norm_Brk

        return (Cs,Bs,As)
    
    elif method == "cp":
        _, (C,B,A) = factors

        As = np.empty(A.shape)
        Bs = np.empty(B.shape)
        Cs = np.empty(C.shape)

        R = A.shape[1]
        K = C.shape[0]
        for r in range(R):
            norm_Ar = tl.norm(A[:, r])
            norm_Br = tl.norm(B[:, r])
            As[:, r] = A[:, r] / norm_Ar
            Bs[:, r] = B[:, r] / norm_Br
            Cs[:, r] = C[:, r] * norm_Ar * norm_Br
            
        return _, (Cs,Bs,As)

def fit_CP_(args):

    dataset_name = args["dataset_name"]
    method = args["method"]
    rank = args["rank"]
    tensor = args["tensor"]
    init_no = args["init_no"]
    mask = args["mask"]
    l2_reg = args["l2_reg"]
 
    rows = []

    try:

        data = deepcopy(tensor)

        start = time.time()

        factors, errors = parafac(data,
                                  rank = int(rank[1:]),
                                  init = 'random',
                                  mask = mask,
                                  cvg_criterion = 'rec_error',
                                  tol = 1e-8,
                                  l2_reg = l2_reg,
                                  normalize_factors = False,
                                  n_iter_max = 2000,
                                  return_errors = True,
                                  random_state = init_no,
                                  verbose = 100,
                                  linesearch=True)
        end = time.time()

        rows.append(
            {   
                "rank": rank,
                "method": method,
                "dataset_name": dataset_name,
                "l": 0,
                "ridge": l2_reg,
                "init_no": init_no,
                "exit": "OK",
                "exec_time": end - start,
                "factors": factors,
                "rec_errors": errors
            }
        )

        logging.info(f"Completed with init {init_no} ({len(errors)} iters)")

    except Exception as e:

        rows.append(
            {
                "rank": rank,
                "method": method,
                "dataset_name": dataset_name,
                "l": 0,
                "ridge": l2_reg,
                "l_B": 0,
                "init_no": init_no,
                "factors": None,
                "rec_errors": None,
                "exit": e,
                "exec_time": 0
            }
        )

        logging.info(f"Failed with init {init_no} ({e})")

    df = pd.DataFrame(rows)

    path = f"{dataset_name}/{method}/{rank}"

    with open(
        f"analysis_results/models/{path}/inits/init{init_no}.pkl",
        "wb",
    ) as f:
        pickle.dump(df, f)

def fit_PARAFAC2_(args):

    dataset_name = args["dataset_name"]
    method = args["method"]
    rank = args["rank"]
    tensor = args["tensor"]
    init_no = args["init_no"]
    mask = args["mask"]
    l2_reg = args["l2_reg"]
    gl_penalty = args["gl_penalty"]

    rows = []

    try:

        data = deepcopy(tensor)

        if np.isnan(data).any():
            em_ind = True
            regs = [[NonNegativity()],[Parafac2(), GeneralizedL2Penalty(gl_penalty)],[]]
            l2_pen = [l2_reg,0,l2_reg]
        else:
            em_ind = False
            regs = [[NonNegativity()],[Parafac2()],[]]
            l2_pen = [l2_reg,l2_reg,l2_reg]

        print("EM is: ", em_ind)
        start = time.time()

        (weights, (C, B, A)), diagnostics = cmf_aoadmm(
            matrices=data,
            rank=int(re.search(r'\d+', rank).group(0)),
            regs=regs,
            return_errors=True,
            l2_penalty=l2_pen,
            n_iter_max=5000,
            inner_n_iter_max=10,
            feasibility_penalty_scale=10,
            tol=1e-8,
            absolute_tol=1e-8,
            feasibility_tol=1e-6,
            inner_tol=1e-6,
            random_state=init_no,
            em=em_ind, # only set to true when there are missing values!
            verbose=100
        )

        end = time.time()

        feasibility_gaps = []

        for iter_feasibility in diagnostics.feasibility_gaps:
            _ = [
                item
                for sublist in iter_feasibility
                for item in sublist
                if len(sublist) != 0
            ]

            feasibility_gaps.append(_)

        rows.append(
            {
                "method": method,
                "rank": rank,
                "dataset_name": dataset_name,
                "l": 0,
                "ridge": l2_reg,
                "gl_penalty": gl_penalty,
                "init_no": init_no,
                "exit": "OK",
                "exec_time": end - start,
                "factors": (deepcopy(C), deepcopy(B), deepcopy(A)),
                "rec_errors": diagnostics.rec_errors,
                "regularized_loss": diagnostics.regularized_loss,
                "feasibility_gaps": feasibility_gaps
            }
        )

        logging.info(
            f"Completed with init {init_no} ({diagnostics.n_iter} iters)"
        )

    except Exception as e:

        rows.append(
            {
                "method": method,
                "rank": rank,
                "dataset_name": dataset_name,
                "l": 0,
                "l_B": 0,
                "ridge": l2_reg,
                "gl_penalty": gl_penalty,
                "init_no": init_no,
                "factors": None,
                "rec_errors": None,
                "regularized_loss": None,
                "feasibility_gaps": None,
                "exit": e,
                "exec_time": 0
            }
        )

        logging.info(f"Failed with init {init_no} ({e})")

    df = pd.DataFrame(rows)

    path = f'{dataset_name}/{method}/{rank}'
    
    with open(
        f"analysis_results/models/{path}/inits/init{init_no}.pkl",
        "wb",
    ) as f:
        pickle.dump(df, f)

def fit_inits_(args):

    dataset_name = args["dataset_name"]
    method = args["method"]
    rank = args["rank"]
    tensor = args["tensor"]
    split_no = args["split_no"]
    init_no = args["init_no"]
    mask = args["mask"]
    l2_reg = args["l2_reg"]
    gl_penalty = args["gl_penalty"]
    dropped_ids = args["dropped_ids"]

    rows = []

    if method == "cp":

        try:

            data = deepcopy(tensor)

            start = time.time()

            factors, errors = parafac(data,
                                    rank = int(rank[1:]),
                                    init = 'random',
                                    mask = mask,
                                    tol = 1e-8,
                                    l2_reg = l2_reg,
                                    normalize_factors = False,
                                    n_iter_max = 2000,
                                    return_errors = True,
                                    random_state = init_no,
                                    verbose = 100,
                                    linesearch=True)
            end = time.time()

            rows.append(
                {   
                    "rank": rank,
                    "method": method,
                    "dataset_name": dataset_name,
                    "l": 0,
                    "ridge": l2_reg,
                    "split_no": split_no,
                    "init_no": init_no,
                    "exit": "OK",
                    "exec_time": end - start,
                    "factors": factors,
                    "rec_errors": errors,
                    "dropped_ids": dropped_ids
                }
            )

            logging.info(
                f"Completed with init {init_no} ({len(errors)} iters)"
            )

        except Exception as e:

            rows.append(
                {
                    "rank": rank,
                    "method": method,
                    "dataset_name": dataset_name,
                    "l": 0,
                    "l_B": 0,
                    "ridge": l2_reg,
                    "split_no": split_no,
                    "init_no": init_no,
                    "factors": None,
                    "rec_errors": None,
                    "exit": e,
                    "exec_time": 0,
                    "dropped_ids": dropped_ids
                }
            )

            logging.info(f"Failed with init {init_no} ({e})")

    elif method == "parafac2":
        try:

            data = deepcopy(tensor)
            if np.isnan(data).any():
                em_ind = True
                regs = [[NonNegativity()],[Parafac2(), GeneralizedL2Penalty(gl_penalty)],[]]
                l2_pen = [l2_reg,0,l2_reg]
            else:
                em_ind = False
                regs = [[NonNegativity()],[Parafac2()],[]]
                l2_pen = [l2_reg,l2_reg,l2_reg]
            start = time.time() 

            (weights, (C, B, A)), diagnostics = cmf_aoadmm(
                matrices=data,
                rank=int(rank[1]),
                regs=regs,
                return_errors=True,
                l2_penalty=l2_pen,
                n_iter_max=5000,
                inner_n_iter_max=10,
                feasibility_penalty_scale=10,
                tol=1e-8,
                absolute_tol=1e-8,
                feasibility_tol=1e-6,
                inner_tol=1e-6,
                random_state=init_no,
                em=em_ind, # only set to true when there are missing values
                verbose=100
            )

            end = time.time()

            feasibility_gaps = []

            for iter_feasibility in diagnostics.feasibility_gaps:
                _ = [
                    item
                    for sublist in iter_feasibility
                    for item in sublist
                    if len(sublist) != 0
                ]

                feasibility_gaps.append(_)

            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "dataset_name": dataset_name,
                    "l": 0,
                    "ridge": l2_reg,
                    "gl_penalty": gl_penalty,
                    "split_no": split_no,
                    "init_no": init_no,
                    "exit": "OK",
                    "exec_time": end - start,
                    "factors": (deepcopy(C), deepcopy(B), deepcopy(A)),
                    "rec_errors": diagnostics.rec_errors,
                    "regularized_loss": diagnostics.regularized_loss,
                    "feasibility_gaps": feasibility_gaps,
                    "dropped_ids": dropped_ids
                }
            )

            logging.info(
                f"Completed with init {init_no} ({diagnostics.n_iter} iters)"
            )

        except Exception as e:

            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "dataset_name": dataset_name,
                    "l": 0,
                    "l_B": 0,
                    "ridge": l2_reg,
                    "gl_penalty": gl_penalty,
                    "split_no": split_no,
                    "init_no": init_no,
                    "factors": None,
                    "rec_errors": None,
                    "regularized_loss": None,
                    "feasibility_gaps": None,
                    "exit": e,
                    "exec_time": 0,
                    "dropped_ids": dropped_ids
                }
            )

            logging.info(f"Failed with init {init_no} ({e})")


    df = pd.DataFrame(rows)

    path = f"analysis_results/replicability/{dataset_name}/{method}/{rank}/split_{split_no}"
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}", exist_ok=True)

    with open(
        f"{path}/init{init_no}.pkl",
        "wb",
    ) as f:
        pickle.dump(df, f)

def collect_fit(fp, dataset, method):

    if dataset == "COPSAC2010":
        with open(fp+'/data/COPSAC2010_data.pkl', 'rb') as f:
            Data = pickle.load(f)
        tensor, sub_id, tax_id = filter_data(Data, n_time=0, f_threshold=0.1)
        time_labels = ["1wk", "1mth", "1yr", "4yr", "6yr"]
        Metadata = pd.read_csv(fp+'/data/COPSAC2010_metadata.csv')
        Taxonomy = pd.read_table(fp+'/data/COPSAC2010_taxonomy.tsv')

    elif dataset == "FARMM":
        Data = np.load(fp+'/data/FARMM_data.npy')
        Data = np.moveaxis(Data, -1, -2).T # subjects by time by taxa
        tensor = np.delete(Data, 0, axis=1)

    tensor = clr(tensor)
    tensor = norm_tensor(tensor)
    mask = np.isfinite(tensor)

    fit_by_R = []
    dirs = [d for d in os.listdir(f'analysis_results/models/{dataset}/{method}/') if os.path.isdir(os.path.join(f'analysis_results/models/{dataset}/{method}/', d))]
    dirs = [s for s in natsorted(dirs) if '_' not in s] # exclude alternative models
    for r in natsorted(dirs):
        DFbest = pd.read_pickle(f"analysis_results/models/{dataset}/{method}/{r}/best_run.pkl")
        if method == "cp":
            try:
                fit_by_R.append(reconstructed_variance(DFbest.loc['factors'], np.moveaxis(tensor,1,2))*100)
            except Exception as e:
                fit_by_R.append(reconstructed_variance(DFbest.loc['factors'],tensor)*100)
        elif method == "parafac2":
            try:
                fit_by_R.append(get_model_fit(DFbest.loc['factors'], tensor, mask))
            except Exception as e:
                fit_by_R.append(get_model_fit((DFbest.loc['factors'][2], DFbest.loc['factors'][1], DFbest.loc['factors'][0]), tensor, mask))

    with open(f"analysis_results/models/{dataset}/{method}/fit_by_R.pkl", 'wb') as f:
        pickle.dump(fit_by_R, f)
    return fit_by_R

def get_par2taxa_fms(Metadata,dfsplit0,dfsplit1):
    id0 = dfsplit0.dropped_ids
    id1 = dfsplit1.dropped_ids
    ix0 = ~np.isin(Metadata.index, id0)
    ix1 = ~np.isin(Metadata.index, id1)
    keptID0 = Metadata.index[ix0].tolist()
    keptID1 = Metadata.index[ix1].tolist()
    intersect0 = ~np.isin(np.asarray(keptID0), id0 + id1)
    intersect1 = ~np.isin(np.asarray(keptID1), id0 + id1)
    
    fac0 = dfsplit0.factors
    fac1 = dfsplit1.factors
    
    C0 = fac0[0][intersect0,:]
    B0 = np.asarray(fac0[1])[intersect0,:,:]
    A0 = fac0[2]

    C1 = fac1[0][intersect1,:]
    B1 = np.asarray(fac1[1])[intersect1,:,:]
    A1 = fac1[2]

    sample_load0 = np.empty((C0.shape[0],B0[0].shape[0],A0.shape[1]),dtype='float64')
    sample_load1 = sample_load0.copy()
    for r in range(C0.shape[1]):
        for k in range(C0.shape[0]):
            sample_load0[k,:,r] = C0[k,r] * B0[k,:,r]
            sample_load1[k,:,r] = C1[k,r] * B1[k,:,r]
    
    return congruence_coefficient(np.vstack(sample_load0),np.vstack(sample_load1))[0]

def get_par2scaledtime_fms(Metadata,dfsplit0,dfsplit1):
    id0 = dfsplit0.dropped_ids
    id1 = dfsplit1.dropped_ids
    ix0 = ~np.isin(Metadata.index, id0)
    ix1 = ~np.isin(Metadata.index, id1)
    keptID0 = Metadata.index[ix0].tolist()
    keptID1 = Metadata.index[ix1].tolist()
    intersect0 = ~np.isin(np.asarray(keptID0), id0 + id1)
    intersect1 = ~np.isin(np.asarray(keptID1), id0 + id1)
    
    fac0 = dfsplit0.factors
    fac1 = dfsplit1.factors
    
    C0 = fac0[0][intersect0,:]
    B0 = np.asarray(fac0[1])[intersect0,:,:]
    A0 = fac0[2]

    C1 = fac1[0][intersect1,:]
    B1 = np.asarray(fac1[1])[intersect1,:,:]
    A1 = fac1[2]

    sample_load0 = np.empty((C0.shape[0],B0[0].shape[0],A0.shape[1]),dtype='float64')
    sample_load1 = sample_load0.copy()
    for r in range(C0.shape[1]):
        for k in range(C0.shape[0]):
            sample_load0[k,:,r] = C0[k,r] * B0[k,:,r]
            sample_load1[k,:,r] = C1[k,r] * B1[k,:,r]
    
    return congruence_coefficient(np.vstack(sample_load0),np.vstack(sample_load1))[0]

def collect_replicability_results(fp, dataset, method):
    if dataset == "COPSAC2010":
        Metadata = pd.read_csv(fp+'/data/COPSAC2010_metadata.csv')
        Metadata.set_index('Abcno',inplace=True)
        rep_ranges = [(i, i + 10) for i in range(0, 100, 10)]
    else:
        rep_ranges = [(i, i + 5) for i in range(0, 50, 5)]
        Metadata = pd.read_csv(fp+'/data/FARMM_metadata.csv', index_col=[0])
        Metadata = Metadata.groupby('SubjectID').agg({
            'study_day': 'first',
            'study_group': 'first'
        })

    FMS_by_R = dict()
    dirs = [d for d in os.listdir(f'analysis_results/replicability/{dataset}/{method}/') if os.path.isdir(os.path.join(f'analysis_results/replicability/{dataset}/{method}/', d))]

    for rank in natsorted(dirs):
        # print(rank)
        FMS_result = []
        failed_split = []
        for rep_range in rep_ranges:
            DFsplits = pd.DataFrame()
            for rep in range(rep_range[0], rep_range[1]):
                df = pd.DataFrame()
                try:
                    for file_no in os.listdir(f"analysis_results/replicability/{dataset}/{method}/{rank}/split_{rep}/"):
                        with open(f"analysis_results/replicability/{dataset}/{method}/{rank}/split_{rep}/{file_no}",'rb') as f:
                            df_temp = pickle.load(f)
                            df_temp = df_temp[df_temp['exit'] == "OK"]
                            df_temp['iterations'] = df_temp['rec_errors'].apply(lambda x: len(x))
                            if df_temp.shape[0] > 0:
                                df = pd.concat([df_temp,df],ignore_index=True)
                except Exception as e:
                    print(e)

                # filter out run if there's NaN or allzero columns in factors:
                for i in range(df.shape[0]):
                    if np.isnan(df['factors'][i][1][0]).any() | np.isnan(df['factors'][i][1][1]).any() | ~df['factors'][i][1][0].any(axis=0).all():
                        df.drop(i,axis=0, inplace=True)
                        
                df['final_rec_error'] = df['rec_errors'].apply(lambda x: x[-1])
                df['degenerate'] = df['factors'].apply(lambda x: check_degenerate(x, method))

                # discard unfeasible runs
                if method == "parafac2":
                    df['feasibility_gaps_last_min'] = df['feasibility_gaps'].apply(lambda x: max(x[-1]))
                    df['feasibility_gaps_last_min_larger_than_1e-6'] = df['feasibility_gaps_last_min'].apply(lambda x: x > 1e-6)
                    df = df[df['feasibility_gaps_last_min_larger_than_1e-6'] == False]
                    # discard runs that did not converge
                    df = df[df['iterations'] <= 5000]
                else:
                    df = df[df['iterations'] <= 2000]

                # discard degenerate runs
                df = df[df['degenerate'] == False]

                if len(df) < 2:
                    print("TOO FEW CONVERGED RUNS!")
                else:

                    df.sort_values(by='final_rec_error',ascending=True, inplace=True)
                    df = df.iloc[[0]]
                    df['factors'] = df['factors'].apply(lambda x: scale_factors(x, method))
                    DFsplits = pd.concat([DFsplits, df.iloc[0].to_frame().T], ignore_index=True)

            # compute pairwise FMS of splits
            cs = list(itertools.combinations(range(len(DFsplits)), 2))
            for i, pair in enumerate(cs):
                if DFsplits['method'].iloc[0] == 'parafac2':
                    # FMS_result.append(congruence_coefficient(DFsplits['factors'].iloc[pair[0]][2],DFsplits['factors'].iloc[pair[1]][2])[0])
                    fmscb = get_par2scaledtime_fms(Metadata,DFsplits.loc[pair[0]],DFsplits.loc[pair[1]])
                    fmsa = congruence_coefficient(DFsplits['factors'].iloc[pair[0]][2],DFsplits['factors'].iloc[pair[1]][2])[0]
                    FMS_result.append([fmsa, fmscb])
                else:
                    FMS_result.append(factor_match_score(DFsplits['factors'].iloc[pair[0]],DFsplits['factors'].iloc[pair[1]], consider_weights=False,absolute_value=True, skip_mode=0))

        FMS_by_R[rank] = FMS_result
        with open(f"analysis_results/replicability/{dataset}/{method}/FMS_by_R.pkl", 'wb') as f:
            pickle.dump(FMS_by_R, f)

    return FMS_by_R
    
def collect_fmscb_models(fp, dataset, method, rank):
    if dataset == "COPSAC2010":
        Metadata = pd.read_csv(fp+'/data/COPSAC2010_metadata.csv')
        Metadata.set_index('Abcno',inplace=True)
        rep_ranges = [(i, i + 10) for i in range(0, 100, 10)]
    else:
        rep_ranges = [(i, i + 5) for i in range(0, 50, 5)]
        Metadata = pd.read_csv(fp+'/data/FARMM_metadata.csv', index_col=[0])
        Metadata = Metadata.groupby('SubjectID').agg({
            'study_day': 'first',
            'study_group': 'first'
        })

    FMS_by_R = dict()
    dirs = [d for d in os.listdir(f'analysis_results/replicability/{dataset}/{method}/') if os.path.isdir(os.path.join(f'analysis_results/replicability/{dataset}/{method}/', d))]

    FMS_result = []
    failed_split = []
    DFall = pd.DataFrame() # rep*split best runs
    for rep_range in rep_ranges:
        DFsplits = pd.DataFrame()
        for rep in range(rep_range[0], rep_range[1]):
            df = pd.DataFrame()
            try:
                for file_no in os.listdir(f"analysis_results/replicability/{dataset}/{method}/{rank}/split_{rep}/"):
                    with open(f"analysis_results/replicability/{dataset}/{method}/{rank}/split_{rep}/{file_no}",'rb') as f:
                        df_temp = pickle.load(f)
                        df_temp = df_temp[df_temp['exit'] == "OK"]
                        df_temp['iterations'] = df_temp['rec_errors'].apply(lambda x: len(x))
                        if df_temp.shape[0] > 0:
                            df = pd.concat([df_temp,df],ignore_index=True)
            except Exception as e:
                print(e)

            # filter out run if there's NaN or allzero columns in factors:
            for i in range(df.shape[0]):
                if np.isnan(df['factors'][i][1][0]).any() | np.isnan(df['factors'][i][1][1]).any() | ~df['factors'][i][1][0].any(axis=0).all():
                    df.drop(i,axis=0, inplace=True)
                    
            df['final_rec_error'] = df['rec_errors'].apply(lambda x: x[-1])
            df['degenerate'] = df['factors'].apply(lambda x: check_degenerate(x, method))

            # discard unfeasible runs
            if method == "parafac2":
                df['feasibility_gaps_last_min'] = df['feasibility_gaps'].apply(lambda x: max(x[-1]))
                df['feasibility_gaps_last_min_larger_than_1e-6'] = df['feasibility_gaps_last_min'].apply(lambda x: x > 1e-6)
                df = df[df['feasibility_gaps_last_min_larger_than_1e-6'] == False]
                # discard runs that did not converge
                df = df[df['iterations'] <= 5000]
            else:
                df = df[df['iterations'] <= 2000]

            # discard degenerate runs
            df = df[df['degenerate'] == False]

            if len(df) < 2:
                print("TOO FEW CONVERGED RUNS!")
            else:

                df.sort_values(by='final_rec_error',ascending=True, inplace=True)
                df = df.iloc[[0]]
                df['factors'] = df['factors'].apply(lambda x: scale_factors(x, method))
                DFsplits = pd.concat([DFsplits, df.iloc[0].to_frame().T], ignore_index=True)
                DFall = pd.concat([DFall, df.iloc[0].to_frame().T], ignore_index=True)

        # compute pairwise FMS of splits
        cs = list(itertools.combinations(range(len(DFsplits)), 2))
        for i, pair in enumerate(cs):
            if DFsplits['method'].iloc[0] == 'parafac2':
                # FMS_result.append(congruence_coefficient(DFsplits['factors'].iloc[pair[0]][2],DFsplits['factors'].iloc[pair[1]][2])[0])
                fmscb = get_par2scaledtime_fms(Metadata,DFsplits.loc[pair[0]],DFsplits.loc[pair[1]])
                fmsa = congruence_coefficient(DFsplits['factors'].iloc[pair[0]][2],DFsplits['factors'].iloc[pair[1]][2])[0]
                FMS_result.append([fmsa, fmscb])
            else:
                FMS_result.append(factor_match_score(DFsplits['factors'].iloc[pair[0]],DFsplits['factors'].iloc[pair[1]], consider_weights=False,absolute_value=True, skip_mode=0))

    return DFall, FMS_result

def get_scaledtime_factors(DF, Metadata, selected):
    id0 = DF.dropped_ids
    ix0 = ~np.isin(Metadata.index, id0)
    keptID0 = Metadata.index[ix0].tolist()
    submodel_pos = np.isin(np.asarray(keptID0), selected)

    (C,B,A) = DF['factors']
    C = C / la.norm(C, axis=0)
    B = np.asarray(B)
    # scale C and B into CB
    CB = np.empty((B.shape[1],C.shape[1]),dtype='float64')
    for r in range(C.shape[1]):
        # for k in range(C.shape[0]):
        CB[:,r] = C[submodel_pos,r] * B[submodel_pos,:,r]
    
    facsACB = ( 
        (np.array([1.0] * C.shape[1])),
        (A, CB) # A and CB
    )
    return facsACB

def permute_to_ref(DF, reftaxfacs, ref_ix=0):
    
    DF = pd.DataFrame(DF, columns=["facs"])
    out = []
    for index, row in DF.iterrows():
        _, h = congruence_coefficient(reftaxfacs, DF.loc[index,"facs"][1][0])
        A = DF.loc[index,"facs"][1][0][:,h]
        CB = DF.loc[index,"facs"][1][1][:,h]
        # sign ambiguity:
        for r in range(A.shape[1]):
            corr = np.corrcoef(reftaxfacs[:,r], A[:,r])[0,1]
            if corr < 0:
                A[:,r] = A[:,r] * -1
                CB[:,r] = CB[:,r] * -1

        out.append(( 
            (np.array([1.0] * A.shape[1])),
            (A, CB)
        ))
    return out

def create_selected_repli_df(DFall, selected_ids, Metadata, taxfacs, ref_ix=0):

    long_form_results = []
    
    for subject_id in selected_ids:
        mask = DFall['dropped_ids'].apply(lambda id_list: subject_id not in id_list)
        df_filtered = DFall[mask]
        
        if df_filtered.empty:
            print(f"Warning: No data found for subject {subject_id}. Skipping.")
            continue

        factors_series = df_filtered.apply(
            get_scaledtime_factors, 
            args=(Metadata, subject_id),
            axis=1
        )
        
        permuted_factors_list = permute_to_ref(factors_series, taxfacs, ref_ix=ref_ix)
        
        ref_row_full_index = df_filtered.index[ref_ix]
        permuted_model_indices = df_filtered.index.drop(ref_row_full_index)

        for submodel_id, permuted_data in zip(permuted_model_indices, permuted_factors_list):
            
            A = permuted_data[1][0]
            CB = permuted_data[1][1]
            
            n_taxa, n_components = A.shape
            for comp in range(n_components):
                for taxon_idx in range(n_taxa):
                    long_form_results.append({
                        'selected_subject': subject_id,
                        'submodel_id': submodel_id,
                        'factor_type': 'A (Taxa)',
                        'component': comp,
                        'dim_1_idx': taxon_idx,  # Represents taxon index
                        'dim_2_idx': np.nan,     # No second dimension
                        'value': A[taxon_idx, comp]
                    })
            
            n_time, n_components_cb = CB.shape

            for comp in range(n_components_cb):
                for time_idx in range(n_time):
                    long_form_results.append({
                        'selected_subject': subject_id,
                        'submodel_id': submodel_id,
                        'factor_type': 'CB (Time)',
                        'component': comp,
                        'Time': time_idx+1,  # Represents time index
                        'dim_2_idx': np.nan,    # No covariate dimension
                        'value': CB[time_idx, comp]
                    })

    if not long_form_results:
        print("Warning: No results were generated.")
        return pd.DataFrame()
        
    return pd.DataFrame(long_form_results)
