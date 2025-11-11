import os
import numpy as np
import pandas as pd
import matcouply
import pickle
import numpy.linalg as la
import scipy.io as sio
from scipy import stats
from statannotations.Annotator import Annotator
import colorcet as cc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc, ticker
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from cycler import cycler
import itertools
from matcouply.coupled_matrices import cmf_to_tensor
import tensorly as tl
from tensorly import cp_normalize
from natsort import natsorted, natsort_keygen
from functions.aux_funcs import check_degenerate, scale_factors
plt.rc('text.latex', preamble=r'\usepackage{amssymb}')
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:bold'
plt.rcParams['mathtext.bfit'] = 'STIXGeneral:bold'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.rm'] = 'STIXGeneral'
plt.rcParams['mathtext.default'] = 'it'

#############################################

def mm2inch(mm):
    return tuple(0.0393701*x for x in mm)

def fit_rep_plot(dataset, method, axs):
    with open(f"analysis_results/models/{dataset}/{method}/fit_by_R.pkl", 'rb') as f:
        Fit_by_R = pickle.load(f)
        
    with open(f"analysis_results/replicability/{dataset}/{method}/FMS_by_R.pkl", 'rb') as f:
        FMS_by_R = pickle.load(f)
        
    if method == "cp":
        df = pd.DataFrame.from_dict(FMS_by_R, orient='index')
        df = df.transpose()
        dflong = pd.melt(df,var_name="rank")
        dflong = dflong.sort_values(by="rank",key=natsort_keygen())
        dflong["rank"] = dflong["rank"].str.strip("R")
        sns.boxplot(x=dflong["rank"],y=dflong["value"], width=.6,fliersize=0,whis=[10, 90], color='#8da0cb', ax=axs[1])
        sns.stripplot(x=dflong["rank"],y=dflong["value"],size=2,color='k', alpha=0.3, ax=axs[1])
    else:
        dflong = pd.DataFrame.from_dict(FMS_by_R).melt(var_name="rank")
        dflong[["FMS_A", "FMS_CB"]] = pd.DataFrame(dflong['value'].to_list(), columns=['FMS_A','FMS_CB'])
        dflong["rank"] = dflong["rank"].str.strip("R")
        dflong.drop(columns=['value'], inplace=True)
        dflong = dflong.melt(id_vars="rank", var_name="FMS")
        colors = ['#66c2a5', '#e78ac3']
        sns.stripplot(x=dflong["rank"],y=dflong["value"],hue=dflong["FMS"],size=2, palette=colors,dodge=True, alpha=0.3, legend=False, ax=axs[1])
        sns.boxplot(x=dflong["rank"],y=dflong["value"],hue=dflong["FMS"],palette=colors, width=.6,fliersize=0,whis=[10, 90], ax=axs[1],zorder=3)
        handles, _ = axs[1].get_legend_handles_labels()          # Get the artists.
        axs[1].legend(handles, [r"FMS$_{\text{A}}$", r"FMS$_{\text{C*B}}$"], loc="best")

    # sns.boxplot(x=dflong["rank"],y=dflong["value"], width=.6,fliersize=0,whis=[10, 90], color='#8da0cb', ax=axs[1])
    # sns.stripplot(x=dflong["rank"],y=dflong["value"],size=2,color='k', alpha=0.3, ax=axs[1])
    axs[1].axhline(0.9, color='#b3b3b3', linestyle='dashed', alpha = 0.4)
    if method == "cp":
        axs[1].set_ylabel(r"FMS$_{\text{AB}}$")
    else:
        axs[1].set_ylabel(r"FMS")

    axs[1].set_ylim(0.5, 1.05)
    axs[0].plot(range(len(Fit_by_R)), Fit_by_R, marker='o', color='#fc8d62', linewidth=1.0)
    axs[0].set_ylim(0, 60)
    axs[0].set_ylabel(r"Fit$\, [\%]$")
    axs[1].set_xlabel(r"R")
    axs[0].set_title(method.upper())
    return


def get_fig2(facs1,facs2,facs3,subject_colors,linestyles, subject_ids):
    R = facs1[1][0].shape[1]

    fig = plt.figure(figsize=mm2inch((178,260)))
    
    gs = gridspec.GridSpec(3, 1, figure=fig,hspace=0.35)

    gs0 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[0], width_ratios=(0.27,0.73),wspace=0.20,hspace=0.14)
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], width_ratios=(0.4,0.6),wspace=0.25,hspace=0.14)
    gs2 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[2], width_ratios=(0.4,0.6),wspace=0.25,hspace=0.14)

    axs0 = gs0.subplots()
    for i in range(R):
        axs0[i,0].bar(np.arange(1,np.shape(facs1[0])[0]+1), facs1[0][:,i], width=0.8,color='#b3b3b3')
        for k in range(np.asarray(facs1[1]).shape[0]):
            axs0[i,1].plot(np.linspace(0, 1, 21), facs1[1][k,:,i], color=subject_colors[i][k], linestyle=linestyles[i][k])
        
    axs1 = gs1.subplots()

    for i in range(R):
        axs1[i,0].bar(np.arange(1,np.shape(facs2[0])[0]+1), facs2[0][:,i], width=0.8,color='#b3b3b3')
        for k in range(np.asarray(facs2[1]).shape[0]):
            axs1[i,1].plot(np.linspace(0, 1, 21), facs2[1][k,:,i], color=subject_colors[i][k], linestyle=linestyles[i][k])
    
    axs2 = gs2.subplots()

    for i in range(R):
        axs2[i,0].bar(np.arange(1,np.shape(facs3[0])[0]+1), facs3[0][:,i], width=0.8,color='#b3b3b3')
        for k in range(np.asarray(facs3[1]).shape[0]):
            axs2[i,1].plot(np.linspace(0, 1, 21), facs3[1][k,:,i], color=subject_colors[i][k], linestyle=linestyles[i][k])

    for ax in np.vstack((axs0, axs1, axs2))[:,0]:
        ax.set_ylim(bottom=-0.05,top=0.4)
        ax.set_xticks([0,18,36])
        ax.set_yticks([0.0,0.15,0.3])

    for ax in np.vstack((axs0, axs1, axs2))[:,1]:
        ax.set_ylim(bottom=-0.02,top=0.16)
        ax.set_xticks([0.0,0.5,1.0])
        ax.set_yticks([0.0,0.06,0.12])

    for i, ax in enumerate(np.vstack((axs0, axs1, axs2)).flatten()):
        ax.set_xmargin(0.05)
        ax.set_ymargin(0.1)
        ax.grid(True,linestyle='--',color='#D6D8CD')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_facecolor("#F7F7F7")
        if i not in [4,5,10,11,16,17]:
            ax.axes.xaxis.set_ticklabels([])

    # legend
    h0 = [Line2D([0], [0], color=c, linestyle=l, label=s, markersize=5) for (c, l, s) in zip(list(dict.fromkeys(subject_colors[0])),list(dict.fromkeys(linestyles[0])),list(dict.fromkeys(subject_ids[0])))]
    c_ = list(dict.fromkeys(subject_colors[1]))
    l_ = linestyles[1][2:9]
    s_ = list(dict.fromkeys(subject_ids[1]))
    h1 = [Line2D([0], [0], color=c, linestyle=l, label=s, markersize=5) for (c, l, s) in zip(c_,l_,s_)]
    h2 = [Line2D([0], [0], color=c, linestyle=l, label=s, markersize=5) for (c, l, s) in zip(list(dict.fromkeys(subject_colors[2])),list(dict.fromkeys(linestyles[2])),list(dict.fromkeys(subject_ids[2])))]
    
    for i in range(3):
        box0A = axs0[i,0].get_position()
        axs0[i,0].set_position([box0A.x0 + 0.04, box0A.y0, box0A.width, box0A.height])
        box0B = axs0[i,1].get_position()
        axs0[i,1].set_position([box0B.x0 + 0.04, box0B.y0, box0B.width * 0.45, box0B.height])

        box1A = axs1[i,0].get_position()
        axs1[i,0].set_position([box1A.x0 + 0.04, box1A.y0, box1A.width, box1A.height])
        box1B = axs1[i,1].get_position()
        axs1[i,1].set_position([box1B.x0 + 0.04, box1B.y0, box1B.width - 0.07, box1B.height])

        box2A = axs2[i,0].get_position()
        axs2[i,0].set_position([box2A.x0 + 0.04, box2A.y0, box2A.width, box2A.height])
        box2B = axs2[i,1].get_position()
        axs2[i,1].set_position([box2B.x0 + 0.04, box2B.y0, box2B.width - 0.07, box2B.height])

    leg1 = axs0[0,1].legend(title = r"host ID $(k)$", handles=h0, reverse=True, handletextpad=0.1, handlelength=1.4, columnspacing=0.3, fontsize=9, title_fontsize=10, frameon=False, bbox_to_anchor=(0.96,1.36))
    leg2a = axs0[1,1].legend(handles=h1[:1], handletextpad=0.1, handlelength=1.4, columnspacing=0.3, fontsize=9, title_fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.96,1.1))
    leg2b = axs0[1,1].legend(handles=h1[1:],ncol=3, handletextpad=0.1, handlelength=1.4, columnspacing=1, fontsize=9, title_fontsize=8, frameon=False, bbox_to_anchor=(0.96,0.82))
    leg3 = axs0[2,1].legend(handles=h2, handletextpad=0.1, handlelength=1.4, columnspacing=0.3, fontsize=9, title_fontsize=8, frameon=False, bbox_to_anchor=(0.96,1.1))

    # ground truth labels
    tax_lab = "Taxa factors" 
    tax_lab2 = r'$\boldsymbol{a}_r$'
    sctime_lab = "Scaled time factors" 
    sctime_lab2 = r'$\{c_{k,r}[\boldsymbol{b}_k]_r\}_{k=1}^{K}$'

    # CP labels
    tax_lab3 = r'$\hat{\boldsymbol{a}}_r$'
    sctime_lab3 = r'$\{\hat{c}_{k,r}\hat{\boldsymbol{b}}_r\}_{k=1}^{K}$' 

    # PARAFAC2 labels
    tax_lab4 = r'$\hat{\boldsymbol{a}}_r$'
    sctime_lab4 = r'$\{\hat{c}_{k,r}[\hat{\boldsymbol{b}}_k]_r\}_{k=1}^{K}$'

    ylabs = ['Ground truth', 'CP', 'PARAFAC2']

    axs0[0,0].annotate(tax_lab, (0.5, 1.38), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 10)
    axs0[0,1].annotate(sctime_lab, (0.5, 1.38), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 10)
    axs0[0,0].annotate(tax_lab2, (0.5, 1.11), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 11)
    axs0[0,1].annotate(sctime_lab2, (0.5, 1.11), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 11)

    axs1[0,0].annotate(tax_lab3, (0.5, 1.11), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 11)
    axs1[0,1].annotate(sctime_lab3, (0.5, 1.11), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 11)

    axs2[0,0].annotate(tax_lab4, (0.5, 1.11), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 11)
    axs2[0,1].annotate(sctime_lab4, (0.5, 1.11), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 11)

    for i, axs in enumerate([axs0[:,0], axs1[:,0], axs2[:,0]]):
        for ix, ax in enumerate(axs):
            ax.annotate(r"$r={comp}$".replace('comp', str(ix+1)),(-57, 22), xycoords = 'axes points',va = 'center', fontsize = 11)

        axs[1].annotate(ylabs[i], (-68, 30), xycoords = 'axes points', va = 'center', ha='center', fontweight = 'bold',rotation=90, fontsize = 10)
        axs[2].set_xlabel('Taxa',labelpad=2)
    for i, axs in enumerate([axs0[:,1], axs1[:,1], axs2[:,1]]):
        axs[2].set_xlabel('Time',labelpad=2)

    axs0[2,0].set_xlabel('Taxa',labelpad=2)
    axs0[2,1].set_xlabel('Time',labelpad=2)
    
    axs0[0,1].add_artist(leg1)
    axs0[1,1].add_artist(leg2a)
    axs0[1,1].add_artist(leg2b)
    axs0[2,1].add_artist(leg3)

    plt.show()
    return fig

def get_filt_sens_fig(M1, M2, M3, CB1, CB2, CB3, mae1, mae2, max_ixs):

    C1, A1, B1 = M1
    C2, A2, B2 = M2
    C3, A3, B3 = M3

    fig = plt.figure(figsize=mm2inch((178,130)))

    gs = gridspec.GridSpec(2,1, figure=fig, height_ratios=(0.5,0.5), hspace=0.6)
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=(0.4,0.6), wspace=0.3, hspace=0.18)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=(0.7,0.3))

    axs0 = gs0.subplots()
    axs1 = gs1.subplots()

    # scatter
    axs0[0].scatter(np.vstack(CB1)[:,4],np.vstack(CB3)[:,4],s=20, c="#a6cee3",edgecolors='white',linewidth=0.4, zorder=1, label="30%")
    axs0[0].scatter(np.vstack(CB1)[:,4],np.vstack(CB2)[:,4],s=20, c="#1f78b4",edgecolors='white',linewidth=0.4, zorder=3, label="20%")
    axs0[0].axline((0, 0), slope=1, linewidth=0.8, color='#b3b3b3', alpha=0.7)
    timeres = stats.pearsonr(np.vstack(CB1)[:,4],np.vstack(CB2)[:,4]).statistic
    axs0[0].set_aspect('equal', adjustable='box')
    axs0[0].set_xlabel(r'$\text{ref.} \quad c_{{k,5}}[\boldsymbol{b}_k]_5$', fontsize=11)
    axs0[0].set_ylabel(r'$\text{alt.} \quad c_{{k,5}}[\boldsymbol{b}_k]_5$', fontsize=11)
    axs0[0].yaxis.set_major_locator(ticker.MaxNLocator(3))
    axs0[0].xaxis.set_major_locator(ticker.MaxNLocator(3))
    axs0[0].set_anchor("NW")
    # hist
    axs0[1].hist(mae1, alpha=0.7, color='#1f78b4', bins=20, zorder=3, label="20%")
    axs0[1].hist(mae2, histtype='stepfilled', alpha=0.7, color='#a6cee3', bins=20 ,zorder=1, label="30%")
    axs0[1].set_xlabel("Mean absolute error")
    axs0[1].set_ylabel("Frequency")
    axs0[1].yaxis.set_major_locator(ticker.MaxNLocator(3))
    axs0[1].ticklabel_format(axis='x', style='sci', scilimits=(-1,1) ,useMathText=True)
    axs0[1].legend(title="filtering", handletextpad=0.2, columnspacing=1.0, labelspacing=0.2, fontsize=10, frameon=False,title_fontproperties={'size':11,'weight':'bold'})

    # worst scaled times
    fnscaledtimeplot_rep(axs1[0],[C1,A1,B1], 5, max_ixs, 'solid')
    fnscaledtimeplot_rep(axs1[0],[C2,A2,B2], 5, max_ixs, 'dashed')
    fnscaledtimeplot_rep(axs1[0],[C3,A3,B3], 5, max_ixs, 'dotted')
    axs1[0].autoscale()
    axs1[0].yaxis.set_major_locator(ticker.MaxNLocator(3))
    axs1[0].set_ylabel(r'$c_{{k,5}}[\boldsymbol{b}_k]_5$', fontsize=11)
    time_labels = ["1wk", "1mth", "1yr", "4yr", "6yr"]
    axs1[0].xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(time_labels)+1))))
    axs1[0].xaxis.set_ticklabels(time_labels, ha="right",rotation_mode="anchor")
    axs1[0].tick_params(axis = 'x', labelrotation=45)
    axs1[0].set_xlabel("Time")

    # legends
    colors = [plt.cm.Set2(i) for i in range(len(max_ixs))]
    labels1 = ["subject " + chr(ord('A')+j) for j in range(len(max_ixs))]
    linestyles = ["solid", "dashed", "dotted"]
    labels2 = ["10% (ref)", "20%", "30%"]

    h1 = [Line2D([0], [0], color=v, markerfacecolor=v, label=k, lw=4) for k, v in zip(labels1,colors)]
    h2 = [Line2D([0], [0], color='k', markerfacecolor='k', label=i, linestyle=j) for i,j in zip(labels2,linestyles)]

    leg1 = axs1[0].legend(title="subject ID", handles=h1, loc='upper center', bbox_to_anchor=(1.3, 1.1),
                        ncols=1, handlelength=2, handleheight=-0.3, handletextpad=0.4, columnspacing=1.0, labelspacing=0.2, fontsize=10, 
                        title_fontproperties={'size':11,'weight':'bold'}, frameon=False)
    axs1[0].add_artist(leg1)
    axs1[0].legend(title="filtering", handles=h2, loc='upper center', bbox_to_anchor=(1.3, 0.38),
                        ncols=1, handlelength=2, handleheight=-0.3, handletextpad=0.4, columnspacing=1.0, labelspacing=0.2, fontsize=10, 
                        title_fontproperties={'size':11,'weight':'bold'}, frameon=False)
    axs1[1].axis('off')
    # panel labels
    labels = ['i)', 'ii)', 'iii)']

    for ix, ax in enumerate([axs0[0], axs0[1], axs1[0]]):
        ax.annotate(
            labels[ix],
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-3.8, 0.3), textcoords='offset fontsize', fontsize='medium', va='bottom', fontfamily='serif')
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(direction="out", length=2)

    plt.subplots_adjust(right=0.8)
    return fig

def reproducibility_plot(results_df):
    no_of_initializations = results_df.shape[0]

    fig,ax = plt.subplots(1,1,figsize=(5,5))
    fms_matrix = np.zeros((no_of_initializations,no_of_initializations))
    ticks = sorted(results_df['final_rec_error'].tolist()) # ascending order
    ticks = [str(round(x,6)) for x in ticks]
    results_df = results_df.sort_values(by='final_rec_error',ascending=True)

    for i in range(no_of_initializations):
        for j in range(no_of_initializations):
            if results_df['method'].iloc[i] == 'parafac2':
                fms_matrix[i,j] = get_fms(results_df['factors'].iloc[i],results_df['factors'].iloc[j])
            else:
                fms_matrix[i,j] = factor_match_score(results_df['factors'].iloc[i],results_df['factors'].iloc[j],consider_weights=False,absolute_value=True)

    ax.imshow(fms_matrix,vmin=0,vmax=1)
    ax.set_yticks(ticks=range(no_of_initializations),labels=ticks,fontsize=4)

    return

def select_taxa_toplot(Model,Taxonomy,c1,c2,nspec,Model2=None):
    R = Model[0].shape[1]
    component1 = c1-1
    component2 = c2-1

    # taxa loadings
    taxa_load = Model[1]

    # taxonmic ranks for plotting
    taxa_load = pd.DataFrame(taxa_load)

    taxa_load.columns = ['comp' + str(x+1) for x in range(R)]
    comp1 = taxa_load.columns[component1]
    Tax = pd.concat([Taxonomy, taxa_load],axis=1)
    top1 = Tax.sort_values(by=comp1,ascending=False, key=abs).head(nspec)

    if Model2 is not None:
        taxa_load2 = Model2[1]
        taxa_load2 = pd.DataFrame(taxa_load2)
        comp2 = taxa_load2.columns[component2]
        Tax2 = pd.concat([Taxonomy, taxa_load2],axis=1)
        top2 = Tax2.sort_values(by=comp2,ascending=False, key=abs).head(nspec)
    else:
        comp2 = taxa_load.columns[component2]
        top2 = Tax.sort_values(by=comp2,ascending=False, key=abs).head(nspec)
        
    ix1 = top1.index.to_list()
    ix2 = top2.index.to_list()
    top = Tax.loc[np.hstack((ix1,ix2)),'Species']
    if Model2 is not None:
        taxa_to_plot1 = Tax.loc[Tax['Species'].isin(top.to_list()),:].sort_values(by=comp1, key=abs)
        taxa_to_plot2 = Tax2.loc[Tax2['Species'].isin(top.to_list()),:].sort_values(by=comp2, key=abs)
        taxa_to_plot = pd.concat([taxa_to_plot1,taxa_to_plot2]).drop_duplicates(subset=['Species'])
    else:
        taxa_to_plot = Tax.loc[Tax['Species'].isin(top.to_list()),:].sort_values(by=[comp1, comp2], key=abs)

    return taxa_to_plot

def factorplot(Model, var=None, Metadata=None, time_labels=None, splines_on=True, nometa=False):

    # number of components
    R = Model[0].shape[1]

    # subject scores
    subject_score = Model[0]
    if Metadata is not None:
        m = np.empty((subject_score.shape[0],len(np.unique(Metadata[var]))),dtype='bool')
        group_sorted = np.array(m.sum(axis=0)).argsort()[::-1]
        for i in group_sorted:
            m[:,i] = Metadata[var] == Metadata[var].value_counts().index[i]
        
        # unique category labels
        color_labels = Metadata[var].value_counts().index.tolist()

        # map label to RGB
        # rgb_values = mpl.colormaps['Dark2'].resampled(8).colors
        # rgb_values = ['#e6ab02', '#e7298a', '#7570b3'] 
        rgb_values = ['#e6ab02', '#e7298a', '#7570b3'] 
        
        color_map = dict(zip(color_labels, rgb_values))
        # legend
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=5) for k, v in color_map.items()]

    # taxa loadings
    taxa_load = Model[1]

    # time loadings
    time_load = Model[2]

    # sample times
    if time_labels is not None:
        times = time_labels
    else:
        times = ['1wk', '1mth', '1yr', '4yr', '6yr']

    fig, axs = plt.subplots(R,3, sharex="col", figsize=(8,6), width_ratios=[0.7,1.5,0.7])

    for i in range(R):
        axs[i,0].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
        if nometa == True:
            axs[i,0].scatter(np.arange(1,np.shape(subject_score)[0]+1), subject_score[:,i],
                            s=20, c='#8da0cb',edgecolors='white',linewidth=0.4)            
        else:
            axs[i,0].scatter(np.arange(1,np.shape(subject_score)[0]+1), subject_score[:,i],
                            s=20, c=Metadata[var].astype(str).map(color_map),edgecolors='white',linewidth=0.4)
        axs[i,0].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[i,0].yaxis.set_major_locator(ticker.MaxNLocator(4))

        axs[i,1].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
        axs[i,1].bar(np.arange(1,np.shape(taxa_load)[0]+1), taxa_load[:,i], width=1.2, color='#cb8da0')
        axs[i,1].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[i,1].yaxis.set_major_locator(ticker.MaxNLocator(4))

        if  time_load.shape[0] == subject_score.shape[0]:
            line_segments = []
            for j in range(len(np.unique(Metadata[var]))):
                line_segments.append(LineCollection([np.column_stack([list(range(1,time_load[0].shape[0]+1)), y]) for y in time_load[m[:,j],:,i]],
                                    colors=color_map[Metadata[var].value_counts().index[j]],
                                    path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()], alpha=0.3))
                line_segments[j].set_array(list(range(1,time_load[0].shape[0]+1)))
                axs[i,2].add_collection(line_segments[j])
        else:
            axs[i,2].plot(np.arange(1,np.shape(time_load)[0]+1), time_load[:,i],linewidth=2, color='#8dcbb8') 
        axs[i,2].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)    
        axs[i,2].xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(times)+1))))
        if len(times) > 5:
            axs[i,2].xaxis.set_major_locator(ticker.MaxNLocator(5))
        else:
            axs[i,2].xaxis.set_major_formatter(ticker.FixedFormatter(times))
        axs[i,2].yaxis.set_major_locator(ticker.MaxNLocator(4))

    for ax in axs.flatten():
        ax.tick_params(direction="out", length=2)

    for ax in axs[:,1]:  
        ax.set_xmargin(0.02)
        ax.set_ymargin(0.1)

    for ax in axs[:,[0,2]].flatten():
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)

    if splines_on == False:
        for i, ax in enumerate(axs.flatten()):
            if ax in axs.flatten()[-3:]:  
                ax.spines[['right', 'top']].set_visible(False)
            else:
                ax.spines[['right', 'bottom', 'top']].set_visible(False)
                ax.get_xaxis().set_visible(False)

    # axis labels
    axs[max(range(R)),0].set_xlabel("Subject")
    axs[max(range(R)),1].set_xlabel("ASV")
    if type(time_labels[0]) == str:
        axs[max(range(R)),2].tick_params(axis = 'x', labelrotation=40)
    else:
        axs[max(range(R)),2].set_xlabel("Study day")
    axs[max(range(R)),0].get_xaxis().set_ticks([])
    axs[max(range(R)),1].get_xaxis().set_ticks([])

    mode = [r'$\boldsymbol{c}_',r'$\boldsymbol{a}_',r'$\boldsymbol{b}_']
    comp = [str(x)+'$' for x in range(1,R+1)]
    label = tuple(x+y for y in comp for x in mode)
    for ix, ax in enumerate(axs.flatten()):
        ax.set_ylabel(f"{label[ix]}",fontsize=10,labelpad=-1)

    axs[0,0].set_title('Subject loadings', y=1.4)
    axs[0,1].set_title('Taxa loadings', y=1.4)
    axs[0,2].set_title('Time loadings', y=1.4)
    fig.align_ylabels()
    fig.subplots_adjust(wspace=0.5,hspace=0.3)
    if nometa==True:
        pass
    else:
        fig.legend(title=var, handles=handles, bbox_to_anchor=(0.06, 0.95+(np.ceil(len(color_labels)/2)-1)*0.025), 
                ncols=3, handleheight=0.4, handletextpad=0.0, columnspacing=0.4, fontsize=8, 
                title_fontsize=8, frameon=False, loc='outside upper left')

    return fig

def scaledfactorplot(Model, Metadata, var, time_labels, splines_on=True, individual=True):

    # number of components
    R = Model[0].shape[1]

    # subject scores
    subject_score = Model[0]
    m = np.empty((subject_score.shape[0],len(np.unique(Metadata[var]))),dtype='bool')
    group_sorted = np.array(m.sum(axis=0)).argsort()[::-1]
    for i in group_sorted:
        m[:,i] = Metadata[var] == Metadata[var].value_counts().index[i]
    color_labels = Metadata[var].value_counts().index.tolist()
    # taxa loadings
    taxa_load = Model[1]

    # time loadings
    time_load = Model[2]

    # fix sign in time mode to be positice
    for i in range(R):
        if np.mean(time_load[:,i]) < 0:
            time_load[:,i] = time_load[:,i] * -1
            subject_score[:,i] = subject_score[:,i] * -1

    # sample loadings (subject_scores * time_load)
    if (individual==False) and (time_load.shape[0] != subject_score.shape[0]):
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0],R),dtype='float64')
        for r in range(R):
            for k in range(subject_score.shape[0]):
                sample_load[k,:,r] = subject_score[k,r] * time_load[:,r]

        sample_load_med = np.empty((len(np.unique(Metadata[var])),time_load.shape[0],R),dtype='float64')
        sample_load_975 = np.empty(shape=sample_load_med.shape)
        sample_load_025 = np.empty(shape=sample_load_med.shape)
        sample_load_mean = np.empty((len(np.unique(Metadata[var])),time_load.shape[0],R),dtype='float64')
        sample_load_mSEM = np.empty(shape=sample_load_med.shape)
        sample_load_pSEM = np.empty(shape=sample_load_med.shape)
        for r in range(R):
            for k in group_sorted:
                sample_load_med[k,:,r] = np.median(sample_load[m[:,k],:,r],axis=0)
                sample_load_975[k,:,r] = np.percentile(sample_load[m[:,k],:,r], 75, axis=0)
                sample_load_025[k,:,r] = np.percentile(sample_load[m[:,k],:,r], 25, axis=0)

                sample_load_mean[k,:,r] = np.mean(sample_load[m[:,k],:,r],axis=0)
                sample_load_pSEM[k,:,r] = sample_load_mean[k,:,r] + stats.sem(sample_load[m[:,k],:,r])           
                sample_load_mSEM[k,:,r] = sample_load_mean[k,:,r] - stats.sem(sample_load[m[:,k],:,r])

    elif time_load.shape[0] == subject_score.shape[0]:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[1],R),dtype='float64')
        for r in range(R):
            for k in range(subject_score.shape[0]):
                sample_load[k,:,r] = subject_score[k,r] * time_load[k,:,r]   

    else:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0],R),dtype='float64')
        for r in range(R):
            for k in range(subject_score.shape[0]):
                sample_load[k,:,r] = subject_score[k,r] * time_load[:,r]


    # list of RGB triplets
    # rgb_values = mpl.colormaps['Dark2'].resampled(8).colors
    rgb_values = ['#e6ab02', '#e7298a', '#7570b3'] 
    # map label to RGB
    color_map = dict(zip(color_labels, rgb_values))
    linestyles = ['solid','dashed','dashdotted']

    # sample times
    times = time_labels

    fig, axs = plt.subplots(R,2, sharex="col", figsize=(8,6), width_ratios=[1.5,0.7])

    for i in range(R):
        axs[i,0].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
        axs[i,0].bar(np.arange(1,np.shape(taxa_load)[0]+1), taxa_load[:,i], color='#cb8da0')
        axs[i,0].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[i,0].yaxis.set_major_locator(ticker.MaxNLocator(4))

        axs[i,1].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)    
        if (individual == True) and (time_load.shape[0] != subject_score.shape[0]):
            line_segments = []
            for j in range(len(np.unique(Metadata[var]))):
                line_segments.append(LineCollection([np.column_stack([list(range(1,time_load.shape[0]+1)), y]) for y in sample_load[m[:,j],:,i]],
                                    colors=color_map[Metadata[var].value_counts().index[j]],
                                    path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()], 
                                    alpha=0.3))
                line_segments[j].set_array((range(1,time_load[0].shape[0]+1)))
                axs[i,1].add_collection(line_segments[j])
        elif time_load.shape[0] == subject_score.shape[0]:
            line_segments = []
            for j in range(len(np.unique(Metadata[var]))):
                line_segments.append(LineCollection([np.column_stack([list(range(1,time_load[0].shape[0]+1)), y]) for y in sample_load[m[:,j],:,i]],
                                    colors=color_map[Metadata[var].value_counts().index[j]],
                                    path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()]))
                line_segments[j].set_array((range(1,time_load[0].shape[0]+1)))
                axs[i,1].add_collection(line_segments[j])
        else:
            line_segments = []
            for j in range(len(np.unique(Metadata[var]))):
                axs[i,1].plot((range(1,time_load[0].shape[0]+1)), sample_load_mean[j,:,i],
                              color=color_map[Metadata[var].value_counts().index[j]], 
                              )
                axs[i,1].fill_between((range(1,time_load[0].shape[0]+1)), sample_load_mSEM[j,:,i], sample_load_pSEM[j,:,i], 
                                      color=color_map[Metadata[var].value_counts().index[j]], 
                                      alpha=0.3, linewidth=0.0)

        axs[i,1].xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(times)+1))))
        if len(times) > 5:
            axs[i,1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        else:
            axs[i,1].xaxis.set_major_formatter(ticker.FixedFormatter(times))

        axs[i,1].ticklabel_format(axis='y', style='sci', scilimits=(-2,2), useMathText=True)

    for ax in axs.flatten():
        ax.tick_params(direction="out", length=2)

    for ix, ax in enumerate(axs[:,0]):  
        ax.set_xmargin(0.02)
        ax.set_ymargin(0.1)
        ax.set_ylabel(r'$\boldsymbol{a}_{{{component}}}$'.replace('component', str(ix+1)), fontsize=10)

    for ix, ax in enumerate(axs[:,1].flatten()):
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.set_ylabel(r'$c_{{{k,component}}}[\boldsymbol{b}_k]_{{{component}}}$'.replace('component', str(ix+1)), fontsize=10)

    if splines_on == False:
        for i, ax in enumerate(axs.flatten()):
            if ax in axs.flatten()[-2:]:  
                ax.spines[['right', 'top']].set_visible(False)
            else:
                ax.spines[['right', 'bottom', 'top']].set_visible(False)
                ax.get_xaxis().set_visible(False)

    # legend
    handles = [Line2D([0], [0], color=v, markerfacecolor=v, label=k, markersize=5) for k, v in color_map.items()]

    # axis labels
    axs[max(range(R)),0].set_xlabel("ASV")
    if type(time_labels[0]) == str:
        axs[max(range(R)),1].tick_params(axis = 'x', labelrotation=40)
    else:
        axs[max(range(R)),1].set_xlabel("Study day")
    axs[max(range(R)),0].get_xaxis().set_ticks([])
    axs[0,0].set_title('Taxa loadings', y=1.4)
    axs[0,1].set_title('Scaled time loadings', y=1.4)

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.legend(title=var, handles=handles, bbox_to_anchor=(0.66, 0.95+(np.ceil(len(color_labels)/2)-1)*0.025), 
                ncols=3, handlelength=1, handleheight=0.4, handletextpad=0.2, columnspacing=0.7, fontsize=8, 
                title_fontsize=8, frameon=False, loc='outside upper left')

    return fig, axs

from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from cycler import cycler
from matplotlib.lines import Line2D

def scaledfactorplot_plain(Model, time_labels):

    # number of components
    R = Model[0].shape[1]

    # subject scores
    subject_score = Model[0]

    # taxa loadings
    taxa_load = Model[1]

    # time loadings
    time_load = Model[2]

    # fix sign in time mode to be positice
    for i in range(R):
        if np.mean(time_load[:,i]) < 0:
            time_load[:,i] = time_load[:,i] * -1
            subject_score[:,i] = subject_score[:,i] * -1

    # sample loadings (subject_scores * time_load)
    if time_load.shape[0] == subject_score.shape[0]:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[1],R),dtype='float64')
        for r in range(R):
            for k in range(subject_score.shape[0]):
                sample_load[k,:,r] = subject_score[k,r] * time_load[k,:,r]   

    else:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0],R),dtype='float64')
        for r in range(R):
            for k in range(subject_score.shape[0]):
                sample_load[k,:,r] = subject_score[k,r] * time_load[:,r]

    # sample times
    times = time_labels

    fig, axs = plt.subplots(R,2, sharex="col", figsize=mm2inch((172,150)), width_ratios=[1.5,0.7])

    for r in range(R):
        axs[r,0].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
        axs[r,0].bar(np.arange(1,np.shape(taxa_load)[0]+1), taxa_load[:,r], color='#cb8da0')
        axs[r,0].xaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[r,0].yaxis.set_major_locator(ticker.MaxNLocator(4))
        axs[r,1].axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)  
        axs[r,1].plot((range(1,time_load[0].shape[0]+1)), sample_load[:,:,r].T, color="#8dd3c7", path_effects=[pe.Stroke(linewidth=2, foreground='w', alpha=0.3), pe.Normal()])
        axs[r,1].xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(times)+1))))
        if len(times) > 5:
            axs[r,1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        else:
            axs[r,1].xaxis.set_major_formatter(ticker.FixedFormatter(times))

        axs[r,0].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
        axs[r,1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)


    for ax in axs.flatten():
        ax.tick_params(direction="out", length=2)

    for ix, ax in enumerate(axs[:,0]):  
        ax.set_xmargin(0.02)
        ax.set_ymargin(0.1)
        ax.set_ylabel(r'$\boldsymbol{a}_{component}$'.replace("component", str(ix+1)), fontsize=10)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    for ix, ax in enumerate(axs[:,1].flatten()):
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
        ax.set_ylabel(r'$c_{k,component}[\boldsymbol{b}_k]_{component}$'.replace('component', str(ix+1)), fontsize=10)
        
    for i, ax in enumerate(axs.flatten()):
        if ax in axs.flatten()[-2:]:  
            ax.spines[['right', 'top']].set_visible(False)
        else:
            ax.spines[['right', 'bottom', 'top']].set_visible(False)
            ax.get_xaxis().set_visible(False)

    axs[max(range(R)),0].set_xlabel("ASV")
    if type(time_labels[0]) == str:
        axs[max(range(R)),1].tick_params(axis = 'x', labelrotation=40)
    else:
        axs[max(range(R)),1].set_xlabel("Study day")
    axs[max(range(R)),0].get_xaxis().set_ticks([])
    
    axs[0,0].set_title('Taxa loadings', y=1.2)
    axs[0,1].set_title('Scaled time loadings', y=1.2)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    return fig

def fnscaledcomponentplot(axs0, axs1, Model, Metadata, Taxonomy, component, var, time_labels, individual=True, labels=True, subtitle=True, legend=True, legtitleheight=1.35, err="SEM",nspec = 17,multcomptaxa=None,Model2=None,tax_col=None):

    # number of components
    R = Model[0].shape[1]
    component = component-1
    # subject scores
    subject_score = Model[0]

    if len(var) != 2:
        m = np.empty((subject_score.shape[0],len(np.unique(Metadata[var]))),dtype='bool')
        ixs = np.array(m.sum(axis=0)).argsort()[::-1]
        for i in ixs:
            m[:,i] = Metadata[var] == Metadata[var].value_counts().index[i]
    elif len(var) == 2:
        m = np.empty((subject_score.shape[0],len(np.unique(Metadata[var[0]]))*len(np.unique(Metadata[var[1]]))),dtype='bool')
        ixs = Metadata[var].value_counts().index.to_list()
        for ix, i in enumerate(ixs):
            m[:,ix] = (Metadata[var[0]] == i[0]) & (Metadata[var[1]] == i[1])

    # taxa loadings
    taxa_load = Model[1]

    # taxonmic ranks for plotting
    taxa_load = pd.DataFrame(taxa_load)
    taxa_load.columns = ['comp' + str(x+1) for x in range(R)]
    comp = taxa_load.columns[component]
    Tax = pd.concat([Taxonomy, taxa_load],axis=1)
    top = Tax.sort_values(by=comp,ascending=False, key=abs).head(nspec)
    ix = top.index.to_list()
    top = Tax.loc[ix,'Species']
    taxa_to_plot = Tax.loc[Tax['Species'].isin(top.to_list()),:].sort_values(by=comp, key=abs)
    if multcomptaxa is not None:
        if Model2 is not None:
            multicomp_taxa_to_plot = select_taxa_toplot(Model,Taxonomy,multcomptaxa[0],multcomptaxa[1],nspec,Model2=Model2)
        else:
            multicomp_taxa_to_plot = select_taxa_toplot(Model,Taxonomy,multcomptaxa[0],multcomptaxa[1],nspec)
        multicomp_tax_color_labels = multicomp_taxa_to_plot['Genus'].value_counts().index
    # time loadings
    time_load = Model[2]

    print(subject_score[0].shape,time_load[0].shape)
    # fix sign in time mode to be positice
    if np.mean(time_load[:,component]) < 0:
        time_load[:,component] = time_load[:,component] * -1
        subject_score[:,component] = subject_score[:,component] * -1

    # sample loadings (subject_scores * time_load)
    if (individual==False) and (time_load.shape[0] != subject_score.shape[0]):
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[:,component]

        sample_load_M = np.empty((len(ixs),time_load.shape[0]),dtype='float64')
        sample_load_H = np.empty(shape=sample_load_M.shape)
        sample_load_L = np.empty(shape=sample_load_M.shape)
        for k in range(len(ixs)):
            if err == "CI95":
                sample_load_M[k,:] = np.median(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = np.percentile(sample_load[m[:,k],:], 75, axis=0)
                sample_load_L[k,:] = np.percentile(sample_load[m[:,k],:], 25, axis=0)
            elif err == "SEM":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.sem(sample_load[m[:,k],:])           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.sem(sample_load[m[:,k],:])
            elif err == "SD":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)    

    elif (individual==True) and time_load.shape[0] == subject_score.shape[0]:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[1]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[k,:,component]

    elif (individual==False) and time_load.shape[0] == subject_score.shape[0]:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[1]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[k,:,component]   

        sample_load_M = np.empty((len(ixs),time_load.shape[1]),dtype='float64')
        sample_load_H = np.empty(shape=sample_load_M.shape)
        sample_load_L = np.empty(shape=sample_load_M.shape)
        for k in range(len(ixs)):
            if err == "IQR":
                sample_load_M[k,:] = np.median(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = np.percentile(sample_load[m[:,k],:], 75, axis=0)
                sample_load_L[k,:] = np.percentile(sample_load[m[:,k],:], 25, axis=0)
            elif err == "SEM":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.sem(sample_load[m[:,k],:])           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.sem(sample_load[m[:,k],:])
            elif err == "SD":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)    

    else:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[:,component]

    # unique category labels
    color_labels = Metadata[var].value_counts().index

    if len(var) == 2:
        color_labels = ixs
        lab1 = np.asarray(ixs)[:,0]
        lab2 = np.asarray(ixs)[:,1]
    # list of RGB triplets
    # rgb_values = mpl.colormaps['Dark2'].resampled(8).colors
    rgb_values = ['#e6ab02', '#e7298a', '#7570b3']
    if len(var) == 2:
        rgb_values = [rgb_values[i] for i in 1*(np.asarray(ixs)[:,0] != ixs[0][0])]
    # map label to RGB
    color_map = dict(zip(color_labels, rgb_values))

    linestyles = ['solid',(0,(4,2)),'dotted','dashdot']
    if len(var) == 2:
        linestyles = [linestyles[i] for i in 1*(np.asarray(ixs)[:,1] != np.asarray(ixs)[0,1])]
        colmap1 = dict(zip(lab1, rgb_values))
        lsmap1 = dict(zip(lab2, linestyles))

    # Taxa colors
    # unique category labels
    tax_color_labels = taxa_to_plot['Genus'].value_counts().index
    
    # list of RGB triplets
    tax_rgb_values = sns.color_palette(cc.glasbey_light, 20)

    if tax_col == None:
        tax_color_map = dict(zip(tax_color_labels, tax_rgb_values))
    else:
        tax_color_map = tax_col
        tax_color_map = dict((k, tax_color_map[k]) for k in tax_color_labels if k in tax_color_map)
        if multcomptaxa is not None:
            multicomp_color_map = dict((k, tax_col[k]) for k in multicomp_tax_color_labels if k in tax_col)
            handles_tax = [plt.Rectangle((0, 0), 1, 1, color=v, label=k.strip()) 
                           for k, v in sorted(multicomp_color_map.items(), key=lambda item: len(item[0]), reverse=True)]
        else:
            handles_tax = [plt.Rectangle((0, 0), 1, 1, color=v, label=k.strip())
                           for k, v in sorted(tax_color_map.items(), key=lambda item: len(item[0]), reverse=True)]
    tax_colors = [tax_color_map[x] for x in taxa_to_plot['Genus'].to_list()]
    tax_labs = taxa_to_plot['Genus'].values

    # sample times
    times = ['1wk', '1mth', '1yr', '4yr', '6yr']

    axs0.axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
    axs0.bar(np.arange(1,len(taxa_to_plot.index.to_list())+1), taxa_load.iloc[taxa_to_plot.index.to_list(),component], color=tax_colors)
    axs0.xaxis.set_ticks(np.linspace(1,taxa_to_plot.shape[0],taxa_to_plot.shape[0]))
    axs0.xaxis.set_ticklabels([x.strip() for x in taxa_to_plot['Species'].to_list()],ha="right",rotation_mode="anchor")
    axs0.yaxis.set_major_locator(ticker.MaxNLocator(4))

    axs1.axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
    if (individual == True) and (time_load.shape[0] != subject_score.shape[0]):
        line_segments = []
        for j in range(len(np.unique(Metadata[var]))):
            line_segments.append(LineCollection([np.column_stack([np.linspace(1,time_load.shape[1],time_load.shape[1]), y]) for y in sample_load[m[:,j],:]],
                                colors=color_map[Metadata[var].value_counts().index[j]],
                                linestyle=linestyles[j],
                                path_effects=[pe.Stroke(linewidth=2, foreground='w', alpha=1), pe.Normal()], 
                                alpha=1))
            line_segments[j].set_array([1, 2, 3, 4, 5])
            axs1.add_collection(line_segments[j])

    elif (individual == True) and (time_load.shape[0] == subject_score.shape[0]):
        line_segments = []
        for j in range(len(ixs)):
            line_segments.append(LineCollection([np.column_stack([np.linspace(1,sample_load.shape[1],sample_load.shape[1]), y]) for y in sample_load[m[:,j],:]],
                                colors=color_map[color_labels[j]],
                                linestyle=linestyles[j],
                                path_effects=[pe.Stroke(linewidth=2, alpha=1), pe.Normal()], 
                                alpha=1))
            line_segments[j].set_array(np.linspace(1,sample_load.shape[1],sample_load.shape[1]))
            axs1.add_collection(line_segments[j])
            
    elif (individual == False) and (time_load.shape[0] == subject_score.shape[0]):
        line_segments = []
        for j in range(len(ixs)):
            axs1.plot(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_M[j,:],
                            color=color_map[color_labels[j]], 
                            linestyle=linestyles[j],
                            alpha=0.8)
            axs1.fill_between(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_L[j,:], sample_load_H[j,:], 
                                    color=color_map[Metadata[var].value_counts().index[j]], 
                                    alpha=0.3, linewidth=0.0)
    else:
        line_segments = []
        for j in range(len(ixs)):
            axs1.plot(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_M[j,:],
                            color=color_map[color_labels[j]], 
                            linestyle=linestyles[j],
                            alpha=0.8)
            axs1.fill_between(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_L[j,:], sample_load_H[j,:], 
                                    color=color_map[Metadata[var].value_counts().index[j]], 
                                    alpha=0.3, linewidth=0.0)

    axs1.xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(time_labels)+1))))
    if len(time_labels) > 5:
        axs1.xaxis.set_major_locator(ticker.MaxNLocator(5))
    else:
        axs1.xaxis.set_major_formatter(ticker.FixedFormatter(time_labels))

    axs1.yaxis.set_major_locator(ticker.MaxNLocator(3))
    axs0.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    axs1.ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
   
    axs = np.vstack((axs0, axs1)).flatten()
    for ax in axs.flatten():
        ax.tick_params(direction="out", length=2)

    axs0.set_xmargin(0.05)
    axs0.set_ymargin(0.1)
    axs1.set_xmargin(0.1)
    axs1.set_ymargin(0.1)

    for i, ax in enumerate(axs):
        if ax in axs.flatten()[-3:]:  
            ax.spines[['right', 'top']].set_visible(False)
        else:
            ax.spines[['right', 'bottom', 'top']].set_visible(False)
            ax.get_xaxis().set_visible(False)

    # legend
    if len(var) != 2:
        handles_var = [Line2D([0], [0], color=v, markerfacecolor=v, label=k, linestyle=l, markersize=5) for (k, v), l in zip(color_map.items(), linestyles)]
    elif len(var) == 2:
        handles_var1 = [Line2D([0], [0], color=v, markerfacecolor=v, label=k) for (k, v) in colmap1.items()]
        handles_var2 = [Line2D([0], [0], color='black', markerfacecolor='black', label=k, linestyle=v, markersize=5) for (k, v) in lsmap1.items()]

    # axis labels
    axs0.set_ylabel(r'$\boldsymbol{a}_{{{component}}}$'.replace('component', str(component+1)), fontsize=11)

    if len(time_load.shape) == 3: # PARAFAC2
        axs1.set_ylabel(r'$c_{{{k,component}}}[\boldsymbol{b}_k]_{{{component}}}$'.replace('component', str(component+1)), fontsize=11)
    elif len(time_load.shape) == 2: # CP
        axs1.set_ylabel(r"$c_{{{k,component}}}\boldsymbol{b}_{{{component}}}$".replace('component', str(component+1)), fontsize=11)
    
    axs0.get_yaxis().set_label_coords(-0.12,0.5)
    axs1.get_yaxis().set_label_coords(-0.24,0.5)

    if subtitle == True:
        tax_lab = "Taxa loading "
        sctime_lab = "Scaled time loading "
        axs0.annotate(tax_lab, (0.5, 1.4), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 10)
        axs1.annotate(sctime_lab, (0.5, 1.4), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 10)

    if labels==True:
        axs0.set_xlabel("Top 15 ASVs", fontsize=10)
        axs1.set_xlabel("Time", fontsize=10)
        
    if type(time_labels[0]) == str:
        axs1.tick_params(axis = 'x', labelrotation=40)
    else:
        axs1.tick_params(axis = 'x', labelsize=10)
        axs1.set_xlabel("Study day")

    axs0.tick_params(axis = 'x', labelrotation=60,labelsize=9)

    if legend == True:
        axs0.legend(title='Genus', handles=handles_tax, loc='upper center', bbox_to_anchor=(0.5, legtitleheight), 
                    ncols=3, handlelength=1.1, handleheight=0.4, handletextpad=0.2, columnspacing=0.7, labelspacing=0.2, fontsize=9, 
                    title_fontproperties={'size':9,'weight':'bold'}, frameon=False)
        if len(var) != 2:
            axs1.legend(title=var, handles=handles_var, loc='upper center', bbox_to_anchor=(0.5, legtitleheight),
                        ncols=1, handlelength=2, handleheight=0.4, handletextpad=0.2, columnspacing=1.0, labelspacing=0.2, fontsize=9, 
                        title_fontproperties={'size':9,'weight':'bold'}, frameon=False)
        elif len(var) == 2:
            l1 = axs1.legend(title=var[0], handles=handles_var1, loc='upper right', bbox_to_anchor=(1.0, legtitleheight),
                        ncols=1, handlelength=2, handleheight=0.4, handletextpad=0.2, columnspacing=0.7, labelspacing=0.2, fontsize=9, 
                        title_fontproperties={'size':9,'weight':'bold'}, frameon=False)
            axs1.add_artist(l1)
            axs1.legend(title=var[1], handles=handles_var2, loc='upper right', bbox_to_anchor=(1.0, legtitleheight),
                        ncols=1, handlelength=2, handleheight=0.4, handletextpad=0.2, columnspacing=0.7, labelspacing=0.2, fontsize=9, 
                        title_fontproperties={'size':9,'weight':'bold'}, frameon=False)    

    return axs1

def taxaplot(ax, Model, Taxonomy, component, labels=True, subtitle=True, legend=True, tax_abbrev=True, legtitleheight=1.35,nspec = 17,multcomptaxa=None,Model2=None,tax_col=None):
    
    # number of components
    R = Model[0].shape[1]
    component = component-1

    # taxa loadings
    taxa_load = Model[1]

    # taxonmic ranks for plotting
    taxa_load = pd.DataFrame(taxa_load)
    taxa_load.columns = ['comp' + str(x+1) for x in range(R)]
    comp = taxa_load.columns[component]
    Tax = pd.concat([Taxonomy, taxa_load],axis=1)
    top = Tax.sort_values(by=comp,ascending=False, key=abs).head(nspec)
    ix = top.index.to_list()
    top = Tax.loc[ix,'Species']
    taxa_to_plot = Tax.loc[Tax['Species'].isin(top.to_list()),:].sort_values(by=comp, key=abs)
    if multcomptaxa is not None:
        if Model2 is not None:
            multicomp_taxa_to_plot = select_taxa_toplot(Model,Taxonomy,multcomptaxa[0],multcomptaxa[1],nspec,Model2=Model2)
        else:
            multicomp_taxa_to_plot = select_taxa_toplot(Model,Taxonomy,multcomptaxa[0],multcomptaxa[1],nspec)
        multicomp_tax_color_labels = multicomp_taxa_to_plot['Genus'].value_counts().index

    # Taxa colors
    # unique category labels
    tax_color_labels = taxa_to_plot['Genus'].value_counts().index
    
    # list of RGB triplets
    tax_rgb_values = sns.color_palette(cc.glasbey_light, 20)

    if tax_col == None:
        tax_color_map = dict(zip(tax_color_labels, tax_rgb_values))
    else:
        tax_color_map = tax_col
        tax_color_map = dict((k, tax_color_map[k]) for k in tax_color_labels if k in tax_color_map)
        if multcomptaxa is not None:
            multicomp_color_map = dict((k, tax_col[k]) for k in multicomp_tax_color_labels if k in tax_col)
            handles_tax = [plt.Rectangle((0, 0), 1, 1, color=v, label=k.strip()) 
                           for k, v in sorted(multicomp_color_map.items(), key=lambda item: len(item[0]), reverse=True)]
        else:
            handles_tax = [plt.Rectangle((0, 0), 1, 1, color=v, label=k.strip())
                           for k, v in sorted(tax_color_map.items(), key=lambda item: len(item[0]), reverse=True)]
    tax_colors = [tax_color_map[x] for x in taxa_to_plot['Genus'].to_list()]
    tax_labs = taxa_to_plot['Genus'].values

    ax.axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)
    ax.bar(np.arange(1,len(taxa_to_plot.index.to_list())+1), taxa_load.iloc[taxa_to_plot.index.to_list(),component], color=tax_colors)
    ax.xaxis.set_ticks(np.linspace(1,taxa_to_plot.shape[0],taxa_to_plot.shape[0]))
    if tax_abbrev == True:
        taxa_to_plot["Species"] = [f"{x.strip().split(' ')[0]} sp." if len(x.strip())>45 else x.strip() for x in taxa_to_plot['Species']]
        # taxa_to_plot["Species"] = [f"{parts[0]}. sp" if len(parts := name.strip().split()) == 2 else name.strip() for name in taxa_to_plot["Species"]]
        ax.xaxis.set_ticklabels([x for x in taxa_to_plot['Species']],ha="right",rotation_mode="anchor")
    else:
        # ax.xaxis.set_ticklabels([x.strip().split('/')[0] + "sp." for x in taxa_to_plot['Species'].to_list()],ha="right",rotation_mode="anchor")
        ax.xaxis.set_ticklabels([x for x in taxa_to_plot['Species']],ha="right",rotation_mode="anchor")

    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.tick_params(direction="out", length=2)
    ax.set_xmargin(0.05)
    ax.set_ymargin(0.1)

    ax.spines[['right', 'top']].set_visible(False)

    # axis labels
    ax.set_ylabel(r"$\boldsymbol{a}_{{{component}}}$".replace("component", str(component+1)), fontsize=11)

    if subtitle == True:
        tax_lab = "Taxa loading "
        ax.annotate(tax_lab, (0.5, 1.4), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 10)

    if labels==True:
        ax.set_xlabel("Top 15 taxa", fontsize=10)
        
    ax.tick_params(axis = 'x', labelrotation=60,labelsize=9)

    if legend == True:
        ax.legend(title='Genus', handles=handles_tax, loc='upper center', bbox_to_anchor=(0.5, legtitleheight), 
                    ncols=3, handlelength=1.1, handleheight=0.4, handletextpad=0.2, columnspacing=0.7, labelspacing=0.2, fontsize=9, 
                    title_fontproperties={'size':9,'weight':'bold'}, frameon=False)

    return

def fnscaledtimeplot(ax, Model, Metadata, component, var, time_labels, individual=True, labels=True, subtitle=True, legend=True, legtitleheight=1.35,ylabshift=-0.27, err="SEM"):

    # number of components
    R = Model[0].shape[1]
    component = component-1
    subject_score = Model[0]

    m = np.empty((subject_score.shape[0],len(Metadata[var].value_counts().index.to_list())),dtype='bool')
    ixs = Metadata[var].value_counts().index.to_list()
    p = [0,2,1] # permute to have VD-no IAP, VD-IAP,CS as plotting order
    ixs = [ixs[i] for i in p]

    for ix, i in enumerate(ixs):
        m[:,ix] = (Metadata[var[0]] == i[0]) & (Metadata[var[1]] == i[1])

    time_load = Model[2]

    print(subject_score[0].shape,time_load.shape)
    # fix sign in time mode to be positice
    if np.mean(time_load[:,component]) < 0:
        time_load[:,component] = time_load[:,component] * -1
        subject_score[:,component] = subject_score[:,component] * -1

    # sample loadings (subject_scores * time_load)
    if (individual==False) and (time_load.shape[0] != subject_score.shape[0]):
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[:,component]

        sample_load_M = np.empty((len(ixs),time_load.shape[0]),dtype='float64')
        sample_load_H = np.empty(shape=sample_load_M.shape)
        sample_load_L = np.empty(shape=sample_load_M.shape)
        for k in range(len(ixs)):
            if err == "CI95":
                sample_load_M[k,:] = np.median(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = np.percentile(sample_load[m[:,k],:], 75, axis=0)
                sample_load_L[k,:] = np.percentile(sample_load[m[:,k],:], 25, axis=0)
            elif err == "SEM":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.sem(sample_load[m[:,k],:])           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.sem(sample_load[m[:,k],:])
            elif err == "SD":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)    

    elif (individual==True) and time_load.shape[0] == subject_score.shape[0]:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[1]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[k,:,component]

    elif (individual==False) and time_load.shape[0] == subject_score.shape[0]:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[1]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[k,:,component]   

        sample_load_M = np.empty((len(ixs),time_load.shape[1]),dtype='float64')
        sample_load_H = np.empty(shape=sample_load_M.shape)
        sample_load_L = np.empty(shape=sample_load_M.shape)
        for k in range(len(ixs)):
            if err == "IQR":
                sample_load_M[k,:] = np.median(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = np.percentile(sample_load[m[:,k],:], 75, axis=0)
                sample_load_L[k,:] = np.percentile(sample_load[m[:,k],:], 25, axis=0)
            elif err == "SEM":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.sem(sample_load[m[:,k],:])           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.sem(sample_load[m[:,k],:])
            elif err == "SD":
                sample_load_M[k,:] = np.mean(sample_load[m[:,k],:],axis=0)
                sample_load_H[k,:] = sample_load_M[k,:] + stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)           
                sample_load_L[k,:] = sample_load_M[k,:] - stats.tstd(sample_load[m[:,k],:],axis=0,keepdims=True)    

    else:
        sample_load = np.empty((subject_score.shape[0],time_load.shape[0]),dtype='float64')
        for k in range(subject_score.shape[0]):
            sample_load[k,:] = subject_score[k,component] * time_load[:,component]

    # unique category labels
    color_labels = ["VD - no IAP", "VD - IAP", "CS"]
    rgb_values = ['#d9d9d9','#66C2A5','#FC8D62']
    # rgb_values = ['#e7298a','#e6ab02','#666666']
    
    color_map = dict(zip(color_labels, rgb_values))
    linestyles = ['solid','dashdot',(0,(3,3))]
    ax.axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)

    if (individual == True) and (time_load.shape[0] != subject_score.shape[0]):
        line_segments = []
        for j in range(len(ixs)):
            line_segments.append(LineCollection([np.column_stack([np.linspace(1,sample_load.shape[1],sample_load.shape[1]), y]) for y in sample_load[m[:,j],:]],
                                colors=color_map[color_labels[j]],
                                linestyle=linestyles[j],
                                linewidth=1.4,
                                alpha=0.7))
            line_segments[j].set_array(np.linspace(1,sample_load.shape[1],sample_load.shape[1]))
            ax.add_collection(line_segments[j])

    elif (individual == True) and (time_load.shape[0] == subject_score.shape[0]):
        line_segments = []
        for j in range(len(ixs)):
            line_segments.append(LineCollection([np.column_stack([np.linspace(1,sample_load.shape[1],sample_load.shape[1]), y]) for y in sample_load[m[:,j],:]],
                                colors=color_map[color_labels[j]],
                                linestyle=linestyles[j],
                                linewidth=1.4,
                                alpha=0.7))
            line_segments[j].set_array(np.linspace(1,sample_load.shape[1],sample_load.shape[1]))
            ax.add_collection(line_segments[j])
            
    elif (individual == False) and (time_load.shape[0] == subject_score.shape[0]):
        line_segments = []
        for j in range(len(ixs)):
            ax.plot(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_M[j,:],
                            color=color_map[color_labels[j]], 
                            linestyle=linestyles[j],
                            alpha=0.8)
            ax.fill_between(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_L[j,:], sample_load_H[j,:], 
                                    color=color_map[color_labels[j]], 
                                    alpha=0.3, linewidth=0.0)
    else:
        line_segments = []
        for j in range(len(ixs)):
            ax.plot(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_M[j,:],
                            color=color_map[color_labels[j]], 
                            linestyle=linestyles[j],
                            alpha=0.8)
            ax.fill_between(np.linspace(1,sample_load_M.shape[1],sample_load_M.shape[1]), sample_load_L[j,:], sample_load_H[j,:], 
                                    color=color_map[color_labels[j]], 
                                    alpha=0.3, linewidth=0.0)

    ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(time_labels)+1))))
    if len(time_labels) > 5:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    else:
        ax.xaxis.set_ticklabels(time_labels, ha="right",rotation_mode="anchor")

    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1) ,useMathText=True)
    ax.tick_params(direction="out", length=2)
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)

    ax.spines[['right', 'top']].set_visible(False)
    handles_var = [Line2D([0], [0], color=v, markerfacecolor=v, label=k, linestyle=l, markersize=5) for (k, v), l in zip(color_map.items(), linestyles)]

    if len(time_load.shape) == 3: # PARAFAC2
        ax.set_ylabel(r"$c_{{{k,component}}}[\boldsymbol{b}_k]_{{{component}}}$".replace('component', str(component+1)), fontsize=11)
    elif len(time_load.shape) == 2: # CP
        ax.set_ylabel(r"$c_{{{k,component}}}\boldsymbol{b}_{{{component}}}$".replace('component', str(component+1)), fontsize=11)
    
    if subtitle == True:
        sctime_lab = "Scaled time loading "
        ax.annotate(sctime_lab, (0.5, 1.4), xycoords = 'axes fraction', ha = 'center', fontweight = 'bold', fontsize = 10)

    if labels==True:
        ax.set_xlabel("Time", fontsize=10)
        
    if type(time_labels[0]) == str:
        ax.tick_params(axis = 'x', labelrotation=45)
    else:
        ax.tick_params(axis = 'x', labelsize=9)
        ax.set_xlabel("Study day")

    if legend == True:

        ax.legend(title="", handles=handles_var, loc='upper center', bbox_to_anchor=(0.5, legtitleheight),
                    ncols=1, handlelength=3, handleheight=0.4, handletextpad=0.2, columnspacing=1.0, labelspacing=0.2, fontsize=8, 
                    title_fontproperties={'size':9,'weight':'bold'}, frameon=False)
 
    return ixs, m, color_map

def fnscaledtimeplot_rep(ax, Model, component, topn, ls, legend=False):
    R = Model[0].shape[1]
    component = component-1
    subject_score = Model[0]
    time_load = Model[2]

    sample_load = np.empty((subject_score.shape[0],time_load.shape[1]),dtype='float64')
    for k in range(subject_score.shape[0]):
        sample_load[k,:] = subject_score[k,component] * time_load[k,:,component]

    ax.axhline(0, linestyle="dashed", linewidth=0.6, c='k',zorder=0)

    colors = [plt.cm.Set2(i) for i in range(len(topn))]
    linecoll = LineCollection([np.column_stack([np.linspace(1,sample_load.shape[1],sample_load.shape[1]), sample_load[i,:]]) for ix, i in enumerate(topn)], array=range(len(topn)), colors=colors, linestyle=ls)
    ax.add_collection(linecoll)

    return

def subjectsplot(ax, Model, Metadata, component):

    component = component-1
    Metadata[str(component+1)] = Model[0][:,component]
    Metadata["testing"] = Metadata["Delivery mode"].astype(str) + ", " + Metadata["Antibiotics"].astype(str).str.lower()

    x = "testing"
    order=["vaginal, no", "vaginal, yes", "c-section, yes"]
    abbrevs = ["VD - no IAP", "VD - IAP", "  CS    "]
    counts = Metadata["testing"].value_counts().loc[order].to_list()
    
    s = sns.boxplot(data=Metadata, y=str(component+1), order=order, hue = x, hue_order=order, orient='v',
                        x=x,fliersize=0, width=0.4, palette=['#d9d9d9','#66C2A5','#FC8D62'],ax=ax)

    sns.stripplot(Metadata, y=str(component+1), x=x, hue = x,hue_order=order, orient='v', size=2, palette=['#cccccc','#54bb9a','#fc7b49'],ax=ax)
    ax.spines[['right', 'top']].set_visible(False)
    # ax.yaxis.label.set_visible(False)
    ax.set_ylabel(r"$\boldsymbol{c}_component$".replace('component', str(component+1)), fontsize=11)
    pairs=[
        ("vaginal, no", "vaginal, yes"),
        ("vaginal, no", "c-section, yes"),
        ("vaginal, yes", "c-section, yes")
        ]
    ax.tick_params(axis = 'x', labelsize=8, labelrotation=45)
    tkz = ax.get_xticks()
    ax.set_xticks(tkz)
    ax.set_xticklabels([g + "\n(N="+str(counts[i])+")" for i, g in enumerate(abbrevs)], ha="right", va="top", rotation_mode="anchor")
    ax.tick_params(direction="out", length=2)
    ax.set(xlabel=None)

    annotator = Annotator(ax, pairs, data=Metadata, y=str(component+1), order=order, orient='v', hue=x, hue_order=order, 
                        x=x)

    annotator.configure(test='Mann-Whitney', text_format = "simple", show_test_name=False, comparisons_correction="bonferroni", loc='outside',use_fixed_offset = False, text_offset=0.03, line_offset=0.02, line_offset_to_group=0.02, fontsize=8)
    annotator.apply_and_annotate()
    return

def subjectsplotfarmm(ax, Model, Metadata, component):

    component = component-1
    Metadata[str(component+1)] = Model[0][:,component]
    Metadata["testing"] = Metadata["Study group"].astype(str)
    rgb_values = ['#e6ab02', '#e7298a', '#7570b3'] 
    x = "testing"
    order=["Vegan", "Omnivore", "EEN"]
    counts = Metadata["testing"].value_counts().loc[order].to_list()
    
    s = sns.boxplot(data=Metadata, x=x, order=order, hue = x, hue_order=order,
                        y=str(component+1),fliersize=0, width=0.4, palette=rgb_values,ax=ax)

    sns.stripplot(Metadata, x=x, y=str(component+1),hue = x,hue_order=order, size=2, palette=rgb_values,ax=ax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(r"$\boldsymbol{c}_component$".replace('component', str(component+1)), fontsize=11)
    pairs=[
        ("Vegan", "Omnivore"),
        ("Vegan", "EEN"),
        ("Omnivore", "EEN")
        ]
    ax.tick_params(axis = 'y', labelsize=10)
    tkz = ax.get_xticks()
    ax.set_xticks(tkz)
    ax.set_xticklabels([g + "\n(N="+str(counts[i])+")" for i, g in enumerate(order)])
    ax.tick_params(direction="out", length=2)
    ax.set_xlabel("")

    annotator = Annotator(ax, pairs, data=Metadata, x=x, order=order, hue=x, hue_order=order, 
                        y=str(component+1))

    annotator.configure(test='Mann-Whitney', text_format = "simple", show_test_name=False, comparisons_correction="bonferroni", loc='outside', use_fixed_offset = False, text_offset=0.03, line_offset=0.02, line_offset_to_group=0.02, fontsize=8,verbose=0)
    annotator.apply_and_annotate()
    return

def get_figS11(CB1,CB2,CB3):

    fig, ax = plt.subplots(2,5,figsize=mm2inch((200,120)),height_ratios=(0.4,0.6))

    for i in range(5):

        d1 = CB1 - CB2
        d2 = CB1 - CB3
        abs_row_means1 = np.mean(np.abs(d1[:,:,i]), axis=1)
        abs_row_means2 = np.mean(np.abs(d2[:,:,i]), axis=1)

        ax[0,i].hist(abs_row_means1, alpha=0.7, color='#1f78b4', bins=20, zorder=3, label="20%")
        ax[0,i].hist(abs_row_means2, histtype='stepfilled', alpha=0.7, color='#a6cee3', bins=20 ,zorder=1, label="30%")
        ax[0,i].set_title(f"$r = {i+1}$",pad=10, fontsize=11)
        ax[0,i].ticklabel_format(axis='x', style='sci', scilimits=(-1,1) ,useMathText=True)
        ax[0,i].tick_params(direction="out", length=2)
        minB = min(np.vstack(CB1)[:,i].min(),np.vstack(CB2)[:,i].min(),np.vstack(CB3)[:,i].min())
        maxB = max(np.vstack(CB1)[:,i].max(),np.vstack(CB2)[:,i].max(),np.vstack(CB3)[:,i].max())

        timeres1 = stats.pearsonr(np.vstack(CB1)[:,i],np.vstack(CB2)[:,i])
        timeres2 = stats.pearsonr(np.vstack(CB1)[:,i],np.vstack(CB3)[:,i])
        timeres1 = timeres1.statistic
        timeres2 = timeres2.statistic
        
        ax[1,i].scatter(np.vstack(CB1)[:,i],np.vstack(CB3)[:,i],s=20, c="#a6cee3",edgecolors='white',linewidth=0.4, zorder=1, label="30%")
        ax[1,i].scatter(np.vstack(CB1)[:,i],np.vstack(CB2)[:,i],s=20, c="#1f78b4",edgecolors='white',linewidth=0.4, zorder=3, label="20%")
        ax[1,i].axline((0, 0), slope=1, linewidth=0.8, color='#b3b3b3', alpha=0.7)
        ax[1,i].text(0.05, 0.87, f'$r$ = {timeres2:.2f}', fontsize=9, color="#a6cee3", transform=ax[1,i].transAxes, horizontalalignment='left')
        ax[1,i].text(0.05, 0.74, f'$r$ = {timeres1:.2f}', fontsize=9, color="#1f78b4", transform=ax[1,i].transAxes, horizontalalignment='left')
        ax[1,i].set_xlim([minB,maxB]) 
        ax[1,i].set_ylim([minB,maxB])
        ax[1,i].set_aspect('equal')
        ax[1,i].ticklabel_format(axis='both', style='sci', scilimits=(-1,1) ,useMathText=True)
        ax[1,i].tick_params(direction="out", length=2)

    ax[0,2].set_xlabel("Mean absolute error",labelpad=15)
    ax[0,0].set_ylabel("Frequency")
    ax[1,2].set_xlabel(r'$ref. \quad c_{{k,r}}[\boldsymbol{b}_k]_r$',labelpad=15, fontsize=11)
    ax[1,0].set_ylabel(r'$alt. \quad c_{{k,r}}[\boldsymbol{b}_k]_r$', fontsize=11)
    ax[0,4].legend(title="filtering", handletextpad=0.2, columnspacing=1.0, labelspacing=0.2, fontsize=10, frameon=False,title_fontproperties={'size':10,'weight':'bold'}, loc='upper center', bbox_to_anchor=(1.5, -0.2))
    labels = ['i)', 'ii)']
    for ix, ax in enumerate([ax[0,0], ax[1,0]]):
        ax.annotate(
            labels[ix],
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-3.5, 0.2), textcoords='offset fontsize', fontsize='medium', va='bottom', fontfamily='serif')
    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    return fig

def get_fig5(SD_copsac, SD_farmm, df_copsac, df_farmm, selected_ids_copsac, selected_ids_farmm, rgb_values_copsac, rgb_values_farmm):
    
    fig = plt.figure(figsize=mm2inch((178,130)))

    gs = gridspec.GridSpec(2,1, figure=fig, height_ratios=(0.5,0.5), hspace=0.8)
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=(0.4,0.6), wspace=0.45, hspace=0.18)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=(0.4,0.6), wspace=0.45, hspace=0.18)

    axs0 = gs0.subplots()
    axs1 = gs1.subplots()

    ## COPSAC
    axs0[0].hist(SD_copsac[(SD_copsac['component'] == 2) & (SD_copsac['Study group'] == "VD - no IAP")]['mean'].values, alpha=0.7, color=rgb_values_copsac[0], label="VD - no IAP\n(N=188)", bins=20, zorder=3)
    axs0[0].hist(SD_copsac[(SD_copsac['component'] == 2) & (SD_copsac['Study group'] == "VD - IAP")]['mean'].values, alpha=0.7, color=rgb_values_copsac[1], label="VD - IAP\n(N=29)", bins=20, zorder=3)
    axs0[0].hist(SD_copsac[(SD_copsac['component'] == 2) & (SD_copsac['Study group'] == "CS")]['mean'].values, alpha=0.7, color=rgb_values_copsac[2], label="CS\n(N=50)", bins=20, zorder=3)
    axs0[0].axvline(x=0.0027,ymax=1, linestyle="dotted", color="#c0c0c0")
    axs0[0].axvline(x=0.0016,ymax=1, linestyle="dashed", color="#c0c0c0")
    axs0[0].axvline(x=0.0020,ymax=1, linestyle="solid", color="#c0c0c0")
    axs0[0].text(x=0.0027,y=51,s="A",rotation=45, rotation_mode='anchor', color="#c0c0c0")
    axs0[0].text(x=0.0016,y=51,s="B",rotation=45, rotation_mode='anchor', color="#c0c0c0")
    axs0[0].text(x=0.0020,y=51,s="C",rotation=45, rotation_mode='anchor', color="#c0c0c0")
    axs0[0].set_xlabel(r"Avg. standard deviation")
    axs0[0].set_ylabel("Frequency")
    axs0[0].legend(title="", handlelength=0.8,handleheight=0.8, handletextpad=0.2, columnspacing=1.0, labelspacing=0.5, fontsize=9, frameon=False,title_fontproperties={'size':10},loc='upper right', bbox_to_anchor=(1.1,1.3))
    for ix, t in enumerate(axs0[0].legend_.get_texts()):
        t.set_verticalalignment("center")
        axs0[0].legend_.legend_handles[ix].set_y(-1.5)

    g = sns.lineplot(x="Time", y="value", errorbar=("sd",2),
                style="selected_subject", 
                style_order=selected_ids_copsac,
                color = "#c0c0c0",
                data=df_copsac, ax=axs0[1])        

    time_labels = ["1wk", "1mth", "1yr", "4yr", "6yr"]
    axs0[1].legend(title = r'subject id ($k$)', handlelength=1.4, handletextpad=0.2,labelspacing=0.2, fontsize=9,frameon=False,loc='upper right', bbox_to_anchor=(1.0,1.25))
    new_labels = ['A', 'B', 'C']
    for t, l in zip(axs0[1].legend_.texts, new_labels):
        t.set_text(l)
    axs0[1].set_ylabel(r'$c_{k,3}[\boldsymbol{b}_k]_{3}$', fontsize=11)
    axs0[1].set_ylim(-0.02,0.084)
    axs0[1].xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(time_labels)+1))))
    axs0[1].xaxis.set_ticklabels(time_labels, ha="right",rotation_mode="anchor")
    axs0[1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    axs0[1].tick_params(axis = 'x', labelrotation=45)
    axs0[1].set_xlabel("")

    ## FARMM
    axs1[0].hist(SD_farmm[(SD_farmm['component'] == 0) & (SD_farmm['Study group'] == "Vegan")]['mean'].values, alpha=0.7, color=rgb_values_farmm[0], label="Vegan\n(N=10)", bins=5, zorder=3)
    axs1[0].hist(SD_farmm[(SD_farmm['component'] == 0) & (SD_farmm['Study group'] == "Omnivore")]['mean'].values, alpha=0.7, color=rgb_values_farmm[1], label="Omnivore\n(N=10)", bins=10, zorder=2)
    axs1[0].hist(SD_farmm[(SD_farmm['component'] == 0) & (SD_farmm['Study group'] == "EEN")]['mean'].values, alpha=0.7, color=rgb_values_farmm[2], label="EEN\n(N=10)", bins=10, zorder=1)

    axs1[0].axvline(x=0.0060,ymax=0.25, linestyle="dotted", color=rgb_values_farmm[2])
    axs1[0].axvline(x=0.0027,ymax=0.9, linestyle="dashed", color=rgb_values_farmm[2])
    axs1[0].axvline(x=0.0029,ymax=0.9, linestyle="solid", color=rgb_values_farmm[2])
    axs1[0].text(x=0.0061,y=1.5,s="9032",rotation=45, rotation_mode='anchor', color=rgb_values_farmm[2])
    axs1[0].text(x=0.0025,y=5.4,s="9024",rotation=45, rotation_mode='anchor', color=rgb_values_farmm[2])
    axs1[0].text(x=0.0031,y=5.4,s="9003",rotation=45, rotation_mode='anchor', color=rgb_values_farmm[2])

    axs1[0].set_xlabel(r"Avg. standard deviation")
    axs1[0].set_ylabel("Frequency")
    axs1[0].set_ylim((0.0,6))
    axs1[0].legend(title="", handlelength=0.8, handleheight=0.8, handletextpad=0.2, columnspacing=1.0, labelspacing=0.5, fontsize=9, frameon=False, title_fontproperties={'size':10},loc='upper right', bbox_to_anchor=(1.05,1.3))
    for ix, t in enumerate(axs1[0].legend_.get_texts()):
        t.set_verticalalignment("center")
        axs1[0].legend_.legend_handles[ix].set_y(-1.5)

    g = sns.lineplot(x="Time", y="value", errorbar=("sd",2),
                style="selected_subject", 
                style_order=selected_ids_farmm,
                color=rgb_values_farmm[2],
                data=df_farmm, ax=axs1[1])

    axs1[1].legend(title = r'subject id ($k$)', handlelength=1.4, handletextpad=0.2,labelspacing=0.2, fontsize=9,frameon=False,loc='upper right', bbox_to_anchor=(1.0,1.25))
    new_labels = ['9003', '9024', '9032']
    for t, l in zip(axs1[1].legend_.texts, new_labels):
        t.set_text(l)
    axs1[1].set_ylabel(r'$c_{k,1}[\boldsymbol{b}_k]_{1}$', fontsize=11)
    axs1[1].set_xlabel("Study day")
    axs1[1].set_ylim(-0.02,0.084)
    axs1[1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
    axs1[1].xaxis.set_major_locator(ticker.MaxNLocator(5))

    labels = ['i)', 'ii)', 'iii)', 'iv)']
    for ix, ax in enumerate([axs0[0], axs0[1], axs1[0], axs1[1]]):
        ax.annotate(
            labels[ix],
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-3.8, 0.3), textcoords='offset fontsize', fontsize='medium', va='bottom', fontfamily='serif')
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(direction="out", length=2)
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
    plt.savefig("analysis_results/figures/Fig5new.png",dpi=600, bbox_inches='tight',pad_inches=0.03)
    plt.show()
        
    return

def get_sup_rep_plots(fn, SD, df, selected_ids, rgb_values):

    R = len(df["component"].dropna().unique())
    fig = plt.figure(figsize=mm2inch((178,36*R)))
    gs = gridspec.GridSpec(R,2, figure=fig, width_ratios=(0.4,0.6), hspace=0.5, wspace=0.4)
    axs = gs.subplots(sharex='col')

    if R == 5:
        profile_col = "#c0c0c0"
    else:   
        profile_col = rgb_values[2]

    ## COPSAC
    for i in range(R):
        if R == 5:
            axs[i,0].hist(SD[(SD['component'] == i) & (SD['Study group'] == "VD - no IAP")]['mean'].values, alpha=0.7, color=rgb_values[0], label="VD - no IAP (N=188)", bins=20, zorder=3)
            axs[i,0].hist(SD[(SD['component'] == i) & (SD['Study group'] == "VD - IAP")]['mean'].values, alpha=0.7, color=rgb_values[1], label="VD - IAP (N=29)", bins=20, zorder=3)
            axs[i,0].hist(SD[(SD['component'] == i) & (SD['Study group'] == "CS")]['mean'].values, alpha=0.7, color=rgb_values[2], label="CS (N=50)", bins=20, zorder=3)
        else:
            axs[i,0].hist(SD[(SD['component'] == i) & (SD['Study group'] == "Vegan")]['mean'].values, alpha=0.7, color=rgb_values[0], label="Vegan (N=10)", bins=5, zorder=3)
            axs[i,0].hist(SD[(SD['component'] == i) & (SD['Study group'] == "Omnivore")]['mean'].values, alpha=0.7, color=rgb_values[1], label="Omnivore (N=10)", bins=10, zorder=2)
            axs[i,0].hist(SD[(SD['component'] == i) & (SD['Study group'] == "EEN")]['mean'].values, alpha=0.7, color=rgb_values[2], label="EEN (N=10)", bins=10, zorder=1)

        axs[i,0].set_ylabel("Frequency")

        g = sns.lineplot(x="Time", y="value", errorbar=("sd",2),
                    style="selected_subject", 
                    style_order=selected_ids,
                    color= profile_col,
                    data=df[df['component']==i], ax=axs[i,1])
        axs[i,1].set_xlabel("")
        axs[i,1].set_ylabel(r"$c_{{{k,component}}}\boldsymbol{b}_{{{component}}}$".replace('component', str(i+1)), fontsize=11)
        axs[i,1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1), useMathText=True)
        g.legend_.remove()

    axs[0,0].legend(title="", handlelength=0.8, handleheight=0.8, handletextpad=0.2, columnspacing=1.0, labelspacing=0.5, fontsize=9, frameon=False, title_fontproperties={'size':10},loc='upper right', bbox_to_anchor=(1.2,1.6))
    for ix, t in enumerate(axs[0,0].legend_.get_texts()):
        t.set_verticalalignment("center")
        axs[0,0].legend_.legend_handles[ix].set_y(-1.5)

 
    if len(df["Time"].dropna().unique()) == 5:
        axs[4,1].set_xlabel("Time")
        time_labels = ["1wk", "1mth", "1yr", "4yr", "6yr"]
        axs[4,1].xaxis.set_major_locator(ticker.FixedLocator(list(range(1,len(time_labels)+1))))
        axs[4,1].xaxis.set_ticklabels(time_labels, ha="right",rotation_mode="anchor")
        axs[4,1].tick_params(axis = 'x', labelrotation=45)
        axs[4,0].set_xlabel(r"Avg. standard deviation")
        axs[0,1].legend(title = r'subject id ($k$)',handlelength=1.4, handletextpad=0.1,labelspacing=0.2, fontsize=9,frameon=False,loc='lower left', bbox_to_anchor=(0.2,0.45))
        new_labels = ['A', 'B', 'C']
        for t, l in zip(axs[0,1].legend_.texts, new_labels):
            t.set_text(l)
    else:
        axs[2,1].set_xlabel("Study day")
        axs[2,1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axs[2,0].set_xlabel(r"Avg. standard deviation")
        axs[0,1].legend(title = r'subject id ($k$)',handlelength=1.4, handletextpad=0.1,labelspacing=0.2, fontsize=9,frameon=False,loc='lower left', bbox_to_anchor=(0.4,0.6))

    labels = ['i)', 'ii)']
    for ix, ax in enumerate([axs[0,0],axs[0,1]]):
        ax.annotate(
            labels[ix],
            xy=(0, 1), xycoords='axes fraction',
            xytext=(-3.8, 0.3), textcoords='offset fontsize', fontsize='medium', va='bottom', fontfamily='serif')
    for ix, ax in enumerate(axs.flatten()):
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(direction="out", length=2)
        ax.set_xmargin(0.1)
        ax.set_ymargin(0.1)
    plt.savefig("analysis_results/figures/"+fn+".png",dpi=600, bbox_inches='tight',pad_inches=0.03)

    plt.show()

    return