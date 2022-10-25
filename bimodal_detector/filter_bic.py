
###################################################
#
# Script: filter_bic.py
# Author: Irene Unterman
# Description: filter and merging functions
#
# MIT License
#
# Copyright (c) 2022 irene unterman and ben berman

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###################################################
import sys
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools")
from epiread_tools.naming_conventions import *
import numpy as np
import pandas as pd
import gzip
from sklearn.metrics.pairwise import nan_euclidean_distances
from itertools import compress

def calc_euclidean(win1, win2):
    '''
    calculate euclidean distance
    :param win1: n-length vector
    :param win2: n-length vector
    :return: distance
    '''
    return nan_euclidean_distances(win1.reshape(1, -1), win2.reshape(1, -1))[0, 0]

def load_bic_from_file(window_summary):
    '''
    load bic summary from step 1
    :param window_summary: file path
    :return: pandas dataframe
    '''
    with open(window_summary) as infile:
        df = pd.read_csv(infile, sep=TAB, usecols=[0,1,2,3], header=None, names=["chromosome",
            "start", "end", "bic"], dtype={"chromosome":str,"start":'int64', "end":'int64', "bic":'float64'})
    return df

def load_sample_summary(sample_summary):
    '''
    load sample summary from step 1
    :param sample_summary: file path
    :return: pandas dataframe
    '''
    with gzip.open(sample_summary) as infile:
        df = pd.read_csv(infile, sep=TAB, header=None, names=["chromosome",
            "start", "end", "sample", "n_reads", "over_0.9", "under_0.1", "over_0.5", "mean", "stdev"],
                          dtype={"chromosome":str,"start":'int64', "end":'int64'})
    return df

def get_mean_pp_vec_from_grouped(group, samples, metric):
    '''
    extract mean pp vector from pd groupby
    :param group: groupby group object
    :param samples: list of samle_ids to extract
    :return: vector of mean pp
    '''
    return group.reset_index(drop=True).reindex(samples)[metric].values

def get_mean_pp_vec_from_stats(stats, stat_col=4):
    '''
    :param stats: sourcesXwindowsXstats matrix
    :param stat_col: column to keep
    :return: windowXsources
    '''
    #cut out first col, it has ALL samples
    return stats.T[stat_col][:,1:]

def apply_threshold(to_filter, thresh):
    return to_filter<thresh

def filter_list(list_to_filter, bool_filter):
    return list(compress(list_to_filter, bool_filter))

def filter_em_results(em_results, list_filter, indices=False):
    new_dict = {}
    for k, v in em_results.items():
        if indices:
            new_dict[k] = list(map(list(v).__getitem__, list_filter))
        else:
            new_dict[k] = filter_list(v, list_filter)
    return new_dict

def merge_indices(win_list, win_metrics, dist_thresh, pairwise_distances=calc_euclidean, overlap_min =0):
    '''
    check if windows overlap and have a pairwise distance below
    dist_thresh
    :param win_list: list of intervals, ordered by start
    :param win_metrics: list of metrics per win
    :param dist_thresh: maximal distance allowed for merge
    :param pairwise_distances: func, distance for 2 vecs
    :param overlap_min: minimal overlap for merge
    :return: where to cut list in order to merge
    '''
    divs = [0]
    for i, (prev_win, window) in enumerate(zip(win_list, win_list[1:])):
        win_start, win_end = window
        prev_start, prev_end = prev_win
        if not (win_start + overlap_min < prev_end and
                pairwise_distances(win_metrics[i], win_metrics[i-1]) < dist_thresh): #don't merge
            divs.append(i+1)
    return divs+[len(win_list)]


def split_old_and_new_windows(divs):
    '''
    find where new windows were created and where old ones
    were kept
    :param divs: where to cut list in order to merge
    :return: indices of old windows, indices of new windows
    '''
    old_indices = []
    new_to_merge = []
    for x, y in zip(divs, divs[1:]):
        if y-x>1:
            new_to_merge.append((x,y))
        else:
            old_indices.append(x)
    return old_indices, new_to_merge

def merge_windows(win_list, new_to_merge):
    '''
    merge intervals
    :param win_list: list of relative coordinate windows
    :param new_to_merge: indices for merging
    :return: list of merged windows
    '''
    rel_windows = []
    for start_ind, end_ind in new_to_merge:
        intervals = win_list[start_ind:end_ind]
        rel_windows.append((min(intervals, key=lambda x: x[0])[0],
                            max(intervals, key=lambda x: x[1])[1]))
    return rel_windows

def get_bic_scores_pre_merging(bic_list, indices):
    scores = []
    for start_ind, end_ind in indices:
        scores.append(np.min(bic_list[start_ind:end_ind]))
    return np.array(scores)

def unmerging(pre_merge_scores, post_merge_scores, minimal_improvement, to_rerun):
    extra_indices_to_keep = []
    to_unmerge = post_merge_scores > pre_merge_scores * minimal_improvement
    rerun_indices = filter_list(to_rerun,to_unmerge)
    for start, end in rerun_indices:
        extra_indices_to_keep.extend(list(range(start,end)))
    return extra_indices_to_keep, ~to_unmerge

    #these are merged window indices, need to go back to unmerged
    #get mat indices from "to_rerun".
    #add these windows to old for strict and remove their output from new


