import os,sys,tqdm
sys.path.append("../")
import pandas as pd
import pickle
from copy import deepcopy
from aux_funcs import *

dataset = sys.argv[1]
method = sys.argv[2]
rank = sys.argv[3]

if not os.path.exists(f"analysis_results/models/{dataset}/{method}/{rank}"):
    os.mkdir(f"analysis_results/models/{dataset}/{method}/{rank}")

df = pd.DataFrame()

print(f'Collecting results for {dataset}, {method}, {rank}...')

# read all results
try:
    for file_no in os.listdir(f"analysis_results/models/{dataset}/{method}/{rank}/inits/"):
        with open(f"analysis_results/models/{dataset}/{method}/{rank}/inits/{file_no}",'rb') as f:
            df_temp = pickle.load(f)

            df_temp = df_temp[df_temp['exit'] == "OK"]

            df_temp['iterations'] = df_temp['rec_errors'].apply(lambda x: len(x))

            if df_temp.shape[0] > 0:
                df = pd.concat([df_temp,df],ignore_index=True)
except Exception as e:
    print(e)

for i in range(df.shape[0]): # filter out run if there's NaN or allzero columns in factors:
    if np.isnan(df['factors'][i][1][0]).any() | np.isnan(df['factors'][i][1][1]).any() | ~df['factors'][i][1][0].any(axis=0).all():
        df.drop(i,axis=0, inplace=True)

df['factors'] = df['factors'].apply(lambda x: scale_factors(x, method))
df['final_rec_error'] = df['rec_errors'].apply(lambda x: x[-1])
df['degenerate'] = df['factors'].apply(lambda x: check_degenerate(x, method))

print(df.shape)

# with open(f"analysis_results/models/{dataset}/{method}/{rank}/results_full.pkl",'wb') as f: # save all runs
#     pickle.dump(df,f)

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

cols = df.columns.to_list()
filtered_df = df.loc[df["final_rec_error"].idxmin()]

print(filtered_df.shape)

with open(f"analysis_results/models/{dataset}/{method}/{rank}/best_run.pkl",'wb') as f: # save best run
    pickle.dump(filtered_df,f)