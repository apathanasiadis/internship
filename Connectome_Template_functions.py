#!/usr/bin/env python
# coding: utf-8

### Import connectomes and BOLD fMRI signals from mat files and calculate FC and FCD.
##### Imports

# get_ipython().run_line_magic('matplotlib', 'widget')
import glob
import sys
import networkx as nx
import vtk
# get_ipython().run_cell_magic('capture', 'output', "import csv\nimport numpy as np\nimport pandas as pd\nimport time as tm\nimport timeit\n\nimport matplotlib.pyplot as plt\nimport matplotlib.gridspec as grd\nimport mpld3 #interactive plot -> mpld3.enable_notebook()\n\nfrom mpl_toolkits.mplot3d import Axes3D\n# create custom color maps\nfrom matplotlib import cm\nplt.style.use('default')\nfrom matplotlib.colors import ListedColormap, LinearSegmentedColormap\nimport seaborn as sns\n\nimport os\nfrom os import listdir\nimport glob\nfrom tqdm import tqdm\nimport scipy\nimport h5py\n\nfrom scipy.io import loadmat\n\nfrom FCD_sp import getFCD\n\n# from tvb.simulator.lab import *\n\n%matplotlib inline\n")
import csv
import numpy as np
import pandas as pd
import time as tm
import timeit

import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import mpld3 #interactive plot -> mpld3.enable_notebook()
from mpl_toolkits.mplot3d import Axes3D
# create custom color maps
from matplotlib import cm
plt.style.use('default')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns

import os
from os import listdir
import glob
from tqdm import tqdm
import scipy
import h5py
from scipy.io import loadmat
from FCD_sp import getFCD

import sklearn
import sklearn.mixture
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import phate


def subject_data(wd_head, data_directory, numpy_file_type,
                 brain_regions_txt, weights_txt, use_time = True, sampling_rate = 1):
    '''
    Inputs:
    cwd_head: working directory head of path, usually 'os.path.split(os.getcwd())[0]'
    data_directory: after head, path of directory that includes data
    file_types: types of files (e.g. npy or np[yz]) to be loaded (uses np.load)
    brain_regions_txt: txt file of brain regions (e.g. '/region_names.txt')
    weights_txt: txt file of weights (e.g. '/weights.txt')
    use_time: index the dataset by time or by samples. (boolean)
    
    Outputs:
    subj_data: list for different kinds (subect to preprocessing of data) of one subject data.
    Each dataset is a pandas dataframe.
    subj_weights: weights matrix
    '''
    
    # READ DATA
    directory_path = wd_head + data_directory
    numpy_vars = {}
    for np_name in glob.glob(directory_path + '*.' + numpy_file_type):
        numpy_vars[np_name] = np.load(np_name).T
        numpy_vars[os.path.split(np_name)[1]] = numpy_vars.pop(np_name)
    
    # CONSTRUCT DATA LIST FOR 1 SUBJECT
    subj_data_list = list(array for array in numpy_vars.values())
    
    # BRAIN REGIONS txt file processing
    subj_bregions  = open(os.path.split(directory_path)[0] + brain_regions_txt).readlines()
    temp = []
    for br in subj_bregions: 
        updated = br.replace('ctx-','')
        temp.append(updated)
    subj_bregions = []
    for br in temp:
        updated = br.replace('\n','')
        subj_bregions.append(updated)
        
    # WEIGHTS
    subj_weights = np.loadtxt(os.path.split(directory_path)[0] + weights_txt)
    
    # build dataframe for each dataset
    N_samples, N_bregions = subj_data_list[0].shape    
    for d in range(len(subj_data_list)):
        subj_data_list[d] = pd.DataFrame(data=subj_data_list[d],
                                        columns=subj_bregions, index = np.arange(N_samples))
    
    TR = sampling_rate
    if use_time == True:
    # time course of data sampling
        timecourse = np.arange(N_samples) * TR
        for d in range(len(subj_data_list)):   
            subj_data_list[d].index = timecourse
    
    return subj_data_list, subj_weights



# preprocessings = ['dicer', 'gsr', 'orig']
def plot_BOLD_ts(subject_data, preprocessings, h=3,
                 fontsize = '9', figsize = (15,50), save_figure = False):
    '''
    Inputs:
    subject_data: list of dataframes of data
    preprocessings: list of names of preprocessings. They should correspond to list of data for correct output.
    ...
    
    Outputs:
    BOLD ts figure
    '''
           
    fig, ax = plt.subplots(len(preprocessings), 1, sharex=True, figsize = figsize)
    for i in range(len(subject_data)):
        t = subject_data[i].index
        subj_bregions = subject_data[i].columns
        N_bregions = subj_bregions.shape[0]
        ax[i].plot(t, (subject_data[i]/subject_data[i].values.max()*2 + np.array(np.r_[:N_bregions] * h)[None,:]).values,
                   'k', alpha=0.6)
        ax[i].set_yticks(ticks = np.r_[:N_bregions]*3, labels = subj_bregions, fontsize = fontsize)
        ax[i].set_title('preprocessing - '+ preprocessings[i])

    ax[len(subject_data)-1].set_xlabel('Time/Samples', fontsize=15)

    plt.tight_layout()
    if save_figure == True:
        plt.savefig('figures/HCP_vsip/BOLD_ts.png')
    else:
        plt.show()


def plot_carpet(subject_data, preprocessings, h=3,
                 fontsize = '9', figsize = (10,4), save_figure = False):
    '''
    Inputs:
    subject_data: list of dataframes of data
    preprocessings: list of names of preprocessings. They should correspond to list of data for correct output.
    ...
    
    Outputs:
    carpet plot for each kind of data, that is implot of the time series.
    '''
    fig, axes = plt.subplots(len(subject_data), 1, sharex=True, figsize = figsize, squeeze=False)
    for i, ax in enumerate(axes.ravel()):

        ax.imshow(subject_data[i].T, cmap='jet', aspect='auto', interpolation='none')
        ax.set_ylabel('nodes', fontsize=11)
        ax.set_title('carpet plot-'+preprocessings[i])
    ax.set_xlabel('Samples/Time')#, fontsize=11)

    plt.tight_layout()
    if save_figure == True:
        plt.savefig('figures/HCP_vsip/BOLD_ts.png')
    else:
        plt.show()
        

def calc_FC(subject_data, preprocessings, plot = False, figsize = (10,3), save_figure = False):
    '''
    Inputs:
    subject_data: list of dataframes of data
    ...
    
    Outputs:
    subject_FC_list: list of FC matrices of shape (N_regions x N_regions)
    FC correlation plots: if plot == True, it plots the FC correlations plots 
    '''
    
    subj_bregions = subject_data[0].columns
    N_bregions = subj_bregions.shape[0]
    subject_FC_list = [] #np.zeros((len(subject_data), N_bregions, N_bregions))
    for d in range(len(subject_data)):    
        subject_FC_list.append(np.corrcoef(subject_data[d],rowvar=False))
        
    if plot == True:
        fig, ax = plt.subplots(1, len(subject_data), sharey=True, figsize = figsize)
        for i in range(len(subject_data)):
            im_FC = ax[i].imshow(subject_FC_list[i], cmap='jet', aspect='auto', interpolation='none')
            ax[i].set_xlabel('brain regions')#, fontsize=11)
            ax[i].set_title('FC-'+preprocessings[i])
        ax[0].set_ylabel('brain regions)')#, fontsize=11)

        fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes(cax)
        plt.colorbar(im_FC, ax=ax[len(subject_data)-1],)# cax=cbar_ax)
        plt.tight_layout()
        if save_figure == True:
            plt.savefig('figures/HCP_vsip/BOLD_ts.png')
        else:
            plt.show()
            
    return subject_FC_list


def calc_FCs_ut(subject_data, window_length, overlap = 29,
                use_time = True, sampling_rate = 1):
    '''
    Inputs:
    subject_data: list of dataframes of data. Indexes should be in line with use_time
    window_length: in seconds or samples depending on use_time argument
    overlap: overlap of time (or samples) between 2 consecutive windows
    ...
    
    Outputs:
    subject_FCs_ut_list:  list of vectorized upper triangles of FC matrices for each window along the whole time course.
    FCs_time: windowed time
    '''
    if overlap >= window_length:
        overlap = window_length-1
    
    N_samples = subject_data[0].index.shape[0]
    subj_bregions = subject_data[0].columns
    N_bregions = subj_bregions.shape[0]
    TR = sampling_rate
    N_ut = int(N_bregions*(N_bregions-1)/2)
    
    if use_time == True:
        n_windows = int(np.floor((N_samples * TR - window_length)/TR + 1))
        subject_FCs_ut_list = []#np.zeros((len(subject_data), N_ut, n_windows))
        for i in range(len(subject_data)):
            FC_t = np.zeros((n_windows, N_bregions, N_bregions))
            FC_t_ut = np.zeros((N_ut, n_windows))
            for w in range(n_windows):
                tw_min = w * TR
                tw_max = window_length  + w * TR 
        #         print(tw_min, tw_max, subj09_data[i].index.values[(tw_min <= subj09_data[i].index.values) & (subj09_data[i].index.values <= tw_max)])
                FC_t_ut[:, w] = np.corrcoef(subject_data[i].loc[subject_data[i].index.values[(tw_min <= subject_data[i].index.values) & (subject_data[i].index.values <= tw_max)]],
                                            rowvar=False)[np.triu_indices(N_bregions, k = 1)]
                # FC_t_ut[:, w] = FC_t[w,:,:][np.triu_indices(N_bregions, k = 1)]
            subject_FCs_ut_list.append(FC_t_ut.T)
        FCs_time = np.arange(n_windows)*TR + 15  
        
        for i in range(len(subject_data)):
            print('sanity check whether the last 30sec time window captures the last time point:\n',
                  tw_min, '<', subject_data[i].index.values[-1],'<=',tw_max)
        
    else:
        window_steps_size = window_length - overlap
        n_windows = int(np.floor((N_samples - window_length) / window_steps_size + 1))

        # upper triangle indices
        Isupdiag = np.triu_indices(N_bregions, 1)    

        #compute FC for each window
        FC_t_ut = np.zeros((N_ut,n_windows))
        # subject_FCs_ut = np.zeros((len(subject_data), N_ut, n_windows))
        subject_FCs_ut_list = []
        for j in range(len(subject_data)):
            ts = subject_data[j].values
            for i in range(n_windows):
                FCtemp = np.corrcoef(ts[window_steps_size*i:window_length + window_steps_size*i,:].T)
                FC_t_ut[:,i] = FCtemp[Isupdiag]
            subject_FCs_ut_list.append(np.array(FC_t_ut.T))
        FCs_time = np.linspace(start = window_length/2, stop = window_length + (n_windows-1)*window_steps_size, 
                               num = n_windows, dtype = float)

    return subject_FCs_ut_list, FCs_time  


def calc_FCs_stream_variants(subject_data, subject_FCs_ut_list):
    '''
    Inputs:
    subject_data: list of dataframes of data. 
    subject_FCs_ut: output of calc_FCs_ut()[0],
                    ndarray of shape (len(subject_data), int(N_regions*(N_regions-1)/2), FCs_time.shape)
    ...
    
    Outputs:
    data_FCs_stream_list: list of mean coherence of arrays of shape (FCs_time.shape)
    data_FCs_stream_diffs_list: list of arrays of shape (FCs_time.shape - 1)
    '''
    
    data_FCs_stream_list = []
    data_FCs_stream_diffs_list = []
    for i in range(len(subject_data)):
        data_FCs_stream_list.append(subject_FCs_ut_list[i].mean(axis = 1))
        data_FCs_stream_diffs_list.append(np.abs(np.diff(subject_FCs_ut_list[1])).mean(axis = 0))
        
    return data_FCs_stream_list, data_FCs_stream_diffs_list



def calc_FCD(subject_data, subject_FCs_ut_list):
    '''
    Inputs:
    subject_data: list of dataframes of data. 
    subject_FCs_ut_list: output of calc_FCs_ut()[0],
                    ndarray of shape (len(subject_data), int(N_regions*(N_regions-1)/2), FCs_time.shape)
    ...
    
    Outputs:
    subject_FCDs_list: list of FCD correlation matrices of shape (FCs_time.shape, FCs_time.shape)
    '''
    
    subject_FCDs_list = []
    for j in range(len(subject_data)):
        subject_FCDs_list.append(np.corrcoef(subject_FCs_ut_list[j]))
        
    return subject_FCDs_list



def calc_edge_ts(subject_data,):
    '''
    Inputs:
    subject_data: list of dataframes of data (N_samples, N_regions) 
        
    Outputs:
    data_Enm_list: list of edge time series (N_samples, N_regions, N_regions)
    data_Enm_ut_list: list of vectorized upper triangular of edge time series (N_samples,N_bregions*(N_bregions-1)/2)
    '''
    N_samples, N_regions = subject_data[0].shape
    data_Enm_list = []
    data_Enm_ut_list = []
    for i in tqdm(range(len(subject_data))):
        data_z_list = (subject_data[i].values - subject_data[i].values.mean(0)[None,:])/subject_data[i].values.std(0)[None,:]
        data_Enm = np.zeros((N_samples, N_regions, N_regions))
        for k in range((N_regions)):
            for j in range(N_regions):
                data_Enm[:,k,j] = np.multiply(data_z_list[:,k], data_z_list[:,j])
        data_Enm_list.append(data_Enm)
        
        data_Enm_ut = np.zeros((N_samples, int(N_regions*(N_regions-1)/2)))
        for n in range(N_samples):
            data_Enm_ut[n, :] = data_Enm_list[i][n,:,:][np.triu_indices(N_regions, k = 1)]
        data_Enm_ut_list.append(data_Enm_ut)
    
    return data_Enm_list, data_Enm_ut_list #data_Enm_list might be redundant




def calc_RSS(subject_data, subject_Enm_list, percentile=95, coactivation_color = 'r'):
    '''
    Inputs:
    subject_data: list of dataframes of data (N_samples, N_regions) 
    subject_Enm_list: list of 3d matrices of edge ts, of shape (N_samples, N_regions, N_regions)
    percentile: The percentile of RSS for coactivation events 
    coactivation_color: It colors the coactivation events
    
    Outputs:
    data_RSS_list: list of RSS series of shape (N_samples)
    data_RSS_95th_percentile_list: coactivation events threshold
    RSS_colors_strings: list of colorized coactivation series of shape (N_samples)
    '''   
    data_RSS_list = []
    data_RSS_95th_percentile_list = []
    RSS_colors_strings = []
    for i in range(len(subject_data)):
        data_RSS_list.append(np.sqrt(np.sum(np.power(subject_Enm_list[i], 2), axis = (1,2))))
        data_RSS_95th_percentile_list.append(np.percentile(data_RSS_list[i], 95))
        
        RSS_colors = np.zeros(subject_data[i].shape[0])
        RSS_colors[np.where(data_RSS_list[i] > data_RSS_95th_percentile_list[i])[0]] = 1
        RSS_colors_strings.append(RSS_colors)
        RSS_colors_strings[i] = np.char.replace(RSS_colors_strings[i].astype(str), '1.0', 'r')
        RSS_colors_strings[i] = np.char.replace(RSS_colors_strings[i].astype(str), '0.0', 'w')
        
    return data_RSS_list, data_RSS_95th_percentile_list, RSS_colors_strings



def run_KMeans(subject_data, n_clusters, random_state=0):
    '''Function that performs K-means clustering for a given number of clusters
        Inputs: 
        subject_data: list of datasets of shape (n_samples, n_features)
        n_clusters: list of number of clusters to apply K-means
        
        Outputs:
        km_labels_list: list of Cluster labels for each dataset for each n_cluster (K-means prediction),
                        of shape (n_clusters, n_samples)
        silhouette_avg_list: silhouette average scores for each K-means clustering
        idx_best_score_cluster_list: list of idx of cluster with the best average score
    '''
    km_labels_list = []
    silhouette_avg_list = []
    idx_best_score_cluster_list = []
    for i in tqdm(range(len(subject_data))):
        if len(subject_data[i].shape) == 1:
            subject_data[i] = np.expand_dims(subject_data[i], axis = 1)
        km = np.zeros((len(n_clusters), subject_data[i].shape[0]))
        silhouette_avg = np.zeros((len(n_clusters)))
        for j,k in enumerate(n_clusters):
            km[j,:] = KMeans(n_clusters=k, random_state=random_state).fit(subject_data[i]).labels_
            silhouette_avg[j] = silhouette_score(subject_data[i], km[j, :])
        km_labels_list.append(km)
        silhouette_avg_list.append(silhouette_avg) 
        idx_best_score_cluster_list.append(silhouette_avg_list[i].argmax())
        
    return km_labels_list, silhouette_avg_list, idx_best_score_cluster_list


def calc_FCD_states(subject_data, windowed_time, overlap_time,
                    km_labels_list, idx_clusters_list, K_FCDstates_colors):
    '''Function that finds the FCD states using K-means 
        Inputs: 
        windowed_time: time array of the FC streams
        km_label_list: list of arrays of shape (n_clusters, windowed_time.shape[0])
        overlap_time: time that is subtracted from the states boarders to ensure no mixture with neighbour states
        idx_cluster_list: list of indices of clusters from n_clusters list to be considered (usually idx_best_score_cluster_list)
                        its length must be equal to len(km_labels_list)
        
        Outputs:
        kmeans2_df_list: list of dataframes of kmeans clutering of the FCD states w.r.t. time indexing of subject_data
        grouped_kmeans2_df_list: list of grouped items according to labels and colors of kmeans2_df_list dataframes
        kmeans2_FCs_time_df_list: list of dataframes of kmeans clutering of the FCD states w.r.t. windowed time (FCs_time)              
    '''
    
    data2_FCDstates = [[] for _ in range(len(idx_clusters_list))]
    kmeans2_FCDstates = [[] for _ in range(len(idx_clusters_list))]
    data2_FCDstates_ids = [[] for _ in range(len(idx_clusters_list))]

    kmeans2_FCDstates_FCs_time = [[] for _ in range(len(idx_clusters_list))]
    kmeans2_ids_FCs_time = [[] for _ in range(len(idx_clusters_list))]
    
    kmeans2_df_list = []
    kmeans2_FCs_time_df_list = []
    grouped_kmeans2_df_list = []
    for j,k in enumerate(tqdm(idx_clusters_list)):
        t = subject_data[j].index.values
        FCDstate_start_time = np.append(np.zeros(1),
                                        windowed_time[np.where(np.diff(km_labels_list[j][k,:]) != 0)[0]]+overlap_time)[:-1] 
        FCDstate_end_time = windowed_time[np.where(np.diff(km_labels_list[j][k,:]) != 0)[0]] - overlap_time
        FCDstate_time_condition = np.append(np.zeros(1),
                                            windowed_time[np.where(np.diff(km_labels_list[j][k,:]) != 0)[0]]+10)[:-1] < windowed_time[np.where(np.diff(km_labels_list[j][k,:]) != 0)[0]]-10
        FCDstate_start_time = FCDstate_start_time[FCDstate_time_condition]
        FCDstate_end_time  = FCDstate_end_time[FCDstate_time_condition]
        

        for i, start_end in enumerate(zip(FCDstate_start_time, FCDstate_end_time)):
            states = subject_data[j].loc[t[(t>=start_end[0]) & (t<=start_end[1])]]
            temp_states = states.values
            temp_ids = states.index
            temp_kmeans_labels = km_labels_list[j][k, :][np.argwhere([(windowed_time-overlap_time >= start_end[0]) & (windowed_time-overlap_time <= start_end[1])][0]).squeeze()]
            temp_kmeans_ids_FCs_time = np.argwhere([(windowed_time >= start_end[0]) & (windowed_time <= start_end[1])][0]).squeeze()
            temp_kmeans_labels_FCs_time = km_labels_list[j][k, :][temp_kmeans_ids_FCs_time]
            # print(i, start_end, temp_kmeans_labels_FCs_time.shape, temp_kmeans_labels_FCs_time)
            if temp_kmeans_labels.shape == ():
                temp_kmeans_labels = [temp_kmeans_labels]
            if np.all(temp_kmeans_labels == temp_kmeans_labels[0]):# or type(temp_kmeans_labels)==float:
                data2_FCDstates[j].append(temp_states)
                kmeans2_FCDstates[j].append(int(temp_kmeans_labels[0]))
                data2_FCDstates_ids[j].append(temp_ids)
            else:
                raise Exception('error0: different k-menas labels for data in the same supposingly class')
            # print(temp_kmeans_labels_FCs_time.size)
            if temp_kmeans_labels_FCs_time.shape == ():
                temp_kmeans_labels_FCs_time = np.expand_dims(temp_kmeans_labels_FCs_time, axis=0)
            if temp_kmeans_labels_FCs_time.size > 0:
                if np.all(temp_kmeans_labels_FCs_time == temp_kmeans_labels_FCs_time[0]):   
                    kmeans2_FCDstates_FCs_time[j].append(int(temp_kmeans_labels[0]))
                    kmeans2_ids_FCs_time[j].append(temp_kmeans_ids_FCs_time)
                else:
                    raise Exception('error1: different k-menas labels for data in the same supposingly class')
                    
        print('check:', len(kmeans2_FCDstates[j]),'>=',len(kmeans2_FCDstates_FCs_time[j]))
        
        kmeans2_df_list.append(pd.DataFrame(index=subject_data[j].index))
        kmeans2_FCs_time_df_list.append(pd.DataFrame(index=np.arange(windowed_time.shape[0])))
        kmeans2_df_list[j][["kmeans labels"]] = -1
        kmeans2_df_list[j][["kmeans colors"]] = 'grey'
        kmeans2_FCs_time_df_list[j][["kmeans labels"]] = -1
        kmeans2_FCs_time_df_list[j][["kmeans colors"]] = 'grey'
        
        for c in range(len(kmeans2_FCDstates[j])):
            kmeans2_df_list[j].loc[data2_FCDstates_ids[j][c], "kmeans labels"] = kmeans2_FCDstates[j][c]
            # print(kmeans2_FCDstates[j][c])
            kmeans2_df_list[j].loc[data2_FCDstates_ids[j][c], "kmeans colors"] = K_FCDstates_colors[kmeans2_FCDstates[j][c]]
        for c in range(len(kmeans2_FCDstates_FCs_time[j])):
            if type(kmeans2_ids_FCs_time[j][c])!=list and kmeans2_ids_FCs_time[j][c].size == 1:
                kmeans2_ids_FCs_time[j][c] = [kmeans2_ids_FCs_time[j][c]]
            # print(kmeans2_FCDstates_FCs_time.size)
            kmeans2_FCs_time_df_list[j].loc[kmeans2_ids_FCs_time[j][c], "kmeans labels"] = kmeans2_FCDstates_FCs_time[j][c]
            kmeans2_FCs_time_df_list[j].loc[kmeans2_ids_FCs_time[j][c], "kmeans colors"] = K_FCDstates_colors[kmeans2_FCDstates_FCs_time[j][c]]
        kmeans2_df_list[j]['index'] =  t
        grouped_kmeans2_df = kmeans2_df_list[j].groupby(["kmeans labels", "kmeans colors"])["index"].apply(list)
        grouped_kmeans2_df_list.append(grouped_kmeans2_df)
        
        
    return kmeans2_df_list, kmeans2_FCs_time_df_list, grouped_kmeans2_df_list




def PCA_sklearn(data, n_features):
    '''
    Function that performs PCA on the input data, using sklearn
    
    input: (BOLD samples, brain regions)-shaped array of (log transformed?) BOLD time series
    output:
        fraction_variance_explained: (brain regions,)-shaped array with the fraction of variance explained by the individual PCs
        principal_components: (brain regions, brain regions)-shaped array containing the principal components as columns
        pca_data: data projected on the PCs space
    '''

    # INSERT YOUR CODE HERE
    f = PCA(n_components=n_features)
    f = f.fit(data)
    
    fraction_variance_explained = f.explained_variance_ratio_
    principal_components = f.components_
    
    principal_components= principal_components.T #to match it with the PCs extracted from before
    pca_data = f.transform(data)
    
    return fraction_variance_explained, principal_components, pca_data


def create_2D_brain_plot(weights, centres, mappable, threshold=1.0, size_edges=0.5,
                        color_nodes='tomato', edgecolors = 'w',
                        size_nodes = 50.0, alpha_node=1.0,
                       save_file = False):
    """
    print three figure and some text about the connectome for the simulation
    :param param: parameter for the simulation
    :param data_1: plot mean firing rate
    :param result_raw: plot spike trains
    # parameter for the background image
    :param image: the image to ad in background of the graph
    :param alpha_image: the transparency of the background image
    # parameter for edge of the graph
    :param threshold: the threshold for the plotting edge of the connectome
    :param size_edges: size of the line of the edges
    # parameter for nodes of the graph
    :param color_Nest: the color of node simulate with Nest
    :param color_TVB: the color of node simulate with TVB
    :param size_node_TVB: the size of the node simulate with TVB
    :param size_node_Nest: the size of the node simulate with Nest
    :param alpha_node: the transparency of the node
    :param size_neurons: size of the marker for neurons
    :return:
    """
    # weights = np.load(path_weight) # path of weight from Spase-file
    
    
    # delays = np.load(param['param_nest_connection']["path_distance"]) / param['param_nest_connection']["velocity"]
    # ids = param['param_co_simulation']["id_region_nest"]
    nb_regions = weights.shape[0] 

    # plot 2d connectome
    # get position of the node
    # centres = np.loadtxt(path_center) # path of center from Spase-file
    Nposition = np.swapaxes(centres[:2, :], 0, 1)
    # print(Nposition, nb_regions)
    # select edges to show
    weights_threshold = np.copy(weights)
    # weights_threshold[np.where(weights == 0.0)] = np.nan
    weights_threshold[np.where(weights_threshold < threshold)] = 0.0
    # select the color for the nodes
    # color_nodes = [color_TVB for i in range(nb_regions)]    

    # select the size of the nodes
    # size_nodes = np.array([size_node_TVB for i in range(nb_regions)])


    # create the graph and print it
    fig = plt.figure(figsize = (5,6))
    ax = plt.gca()
    G = nx.from_numpy_matrix(weights_threshold)
    nx.draw(G, width=size_edges, pos=Nposition, edge_color='#909089', ax=ax,
            node_color=color_nodes, edgecolors=edgecolors, node_size=size_nodes,
            node_shape='o', alpha=alpha_node)
    plt.colorbar(mappable, ax = ax, shrink = 0.6)
    if save_file == True:
        plt.savefig('figures/presentation020622/6brain_PCAloadings_example.png')
    else:
        plt.show()

