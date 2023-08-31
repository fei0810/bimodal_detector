###################################################
#
# Script: run_em.py
# Author: Irene Unterman
# Description: run EM on large genomic chunk
#
# MIT License
#
# Copyright (c) 2022 irene unterman and ben berman
#
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
# ###################################################

import sys
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools")
sys.path.append("/Users/ireneu/PycharmProjects/bimodal_detector")

from bimodal_detector.expectation_maximization import *
from bimodal_detector.Likelihood_and_BIC import *
from bimodal_detector.runner_utils import format_array
from collections import defaultdict
from epiread_tools.naming_conventions import *
#%%

def run_em(methylation_matrix, windows):
    '''
    run expectation maximization on all windows
    :param windows: list of relative windows to run em on
    :param methylation_matrix: sparse matrix of methylation reads
    :return: em output dict for all eligible windows.
    '''

    methylation_matrix = methylation_matrix.tocsc()
    output = defaultdict(list)
    read_indices = np.arange(methylation_matrix.shape[0])

    for start, stop in windows:
        section, sec_read_ind = clean_section(methylation_matrix, start, stop)
        if not ((section == METHYLATED).any(axis=0) & (section == UNMETHYLATED).any(axis=0)).any():  # uniform data
            continue
        BIC, thetas, probs = run_em_on_window(section.shape[1], section.shape[0],
                                                      section, initial_high, initial_low)
        med_cpg = np.nanmedian(np.nansum((section==METHYLATED)|(section==UNMETHYLATED), axis=1))
        percent_single = np.nansum(np.nansum((section==METHYLATED)|(section==UNMETHYLATED), axis=1)==1)/section.shape[0]
        output["windows"].append((start,stop))
        output["BIC"].append(BIC)
        output["median_cpg"].append(med_cpg)
        output["percent_single"].append(percent_single)
        output["Theta_A"].append(thetas[0])
        output["Theta_B"].append(thetas[1])
        output["Probs"].append(probs)
        output["Indices"].append(sec_read_ind)

    output["BIC"] = np.array(output["BIC"])

    return output

def clean_section(methylation_matrix, start, stop):
    '''
    remove empty rows from section
    :param methylation_matrix: csc matrix
    :param start: index
    :param stop: index
    :return: clean section, filtered row indices
    '''
    section = methylation_matrix[:, start:stop]
    read_indices = np.arange(methylation_matrix.shape[0])
    section.eliminate_zeros()
    row_filter = section.getnnz(1) > 0
    section = section[row_filter].toarray()
    sec_read_ind = read_indices[row_filter]
    return section, sec_read_ind

def get_all_stats(row_filters, pp_vectors, ind_to_source, n_sources, get_pp):
    '''
    get summary statistics on all eligible windows
    :param row_filters: list of arrays with indices of reads per window
    :param pp_vectors: list of arrays with posterior probs of reads per window
    :param ind_to_source: mapping from read index to source ID
    :param source_list: list of source IDs
    :param get_pp: keep posterior probabilities per source in output
    :return: sourcesXwindowsXstats matrix
    '''
    n_windows = len(pp_vectors)
    n_sources = n_sources + 1 #add one for ALL
    n_cols = len(get_source_stats(np.zeros(1)))#length of stats
    output = np.full(fill_value=np.nan, shape=(n_sources, n_windows, n_cols))
    if get_pp:
        all_pp = np.full(fill_value=np.nan, shape=(n_sources, n_windows, 1), dtype=object)
    for i in range(n_windows):
        row_filter = row_filters[i]
        probs = pp_vectors[i]
        sec_sources = np.array([ind_to_source[ind] for ind in row_filter])
        output[ALL][i] = get_source_stats(probs)
        if get_pp:
            all_pp[ALL][i] = format_array(probs)
        for source in set(sec_sources):
            source_filter = np.where(sec_sources == source)[0]
            source_probs = probs[source_filter]
            output[source][i] = get_source_stats(source_probs)
            if get_pp:
                all_pp[source][i] = format_array(source_probs)
    if get_pp:
        output = np.concatenate([output, all_pp], axis=2)
    return output

def get_all_snp_stats(snp_matrix, row_filters, pp_vectors, ind_to_source, get_pp):
    '''
    get summary haplotypes for all eligible windows
    :param snp_matrix: sparse matrix with snp data
    :param row_filters: list of arrays with indices of reads per window
    :param pp_vectors: list of arrays with posterior probs of reads per window
    :param source_list: list of source IDs
    :param get_pp: keep posterior probabilities per source in output
    :return: dict per sample with window id as index and stats as values
    '''
    snp_matrix = snp_matrix.tocsc()
    snp_output = defaultdict(dict)
    n_cols = len(get_source_stats(np.zeros(1))) + 1 # length of stats + haplotype
    if get_pp:
        n_cols += 1

    for i in range(len(pp_vectors)):
        probs = pp_vectors[i]
        row_filter = row_filters[i]
        sec_sources = np.array([ind_to_source(ind) for ind in row_filter])
        sec_snps = snp_matrix[row_filter]
        col_filter = sec_snps.getnnz(0) > 0
        sec_snps = sec_snps[:, col_filter].todense()  # has col data
        rel_snps = np.arange(len(col_filter))[col_filter]
        snp_output["SNP_coords"][i] = rel_snps

        if sec_snps.any():
            snp_output[ALL][i] = summarize_haplotypes(sec_snps, probs, n_cols, get_pp)

        for source in set(sec_sources):
            source_filter = np.where(sec_sources == source)[0]
            source_probs = probs[source_filter]
            if sec_snps[source_filter].any():
                snp_output[source][i] = summarize_haplotypes(sec_snps[source_filter], source_probs,
                                                                         n_cols, get_pp)
    return snp_output

def do_walk_on_list(window_list, window_size, step_size):
    '''
    genome walker for list of windows,
    for example when dist thresh is required
    :param window_list: list of relative (start, end) coordinates
    :param window_size: int, N CpGs per window
    :param step_size: int, step size for genome walker
    :return: generator of windows
    '''
    for win_start, win_end in window_list:
        yield from make_genome_walker(window_size, step_size, win_start, win_end)

def summarize_haplotypes(section_snps, section_probs, n_cols, get_pp):
    '''
    get one row per haplotype in window
    :param section_snps: snp array (reads x snps)
    :param section_probs: EM pp, must match row number in snps
    :param n_cols: output columns
    :param get_pp: add raw posterior probabilities to output
    :return: np array of haplotypes and stats
    '''
    haplotypes, indices = find_haplotypes(section_snps)

    hap_array =  np.zeros(shape=(len(haplotypes),n_cols), dtype=object)
    for i in range(len(haplotypes)):#number of haplotypes
        haplotype = haplotypes[i]
        hap_indices = np.where(indices == i)[0]
        stats = get_source_stats(section_probs[hap_indices], get_pp)
        hap_array[i,:] = haplotype, *stats
    return hap_array

def make_genome_walker(window_size, step_size,min_win, max_win):
    '''
    get generator of slices
    :param window_size: int, N CpGs per window
    :param step_size: int, step size for genome walker
    :param min_win: start index
    :param max_win: stop index
    :return: generator of slices
    '''
    for i in range(min_win, max_win, step_size):
        if i+window_size <= max_win:
            yield i, i+window_size

def get_source_stats(source_probs):
    '''
    summarize probs
    :param source_probs: np array of posterior probabilities
    :param pp: whether to output the raw probabilities
    :return: summary statistics:
    n_reads, state A reads, state B reads, reads > 0.5, mean_pp, stdev
    '''
    stats = np.array([len(source_probs), np.sum(source_probs>upper_conf_thresh),
             np.sum(source_probs < lower_conf_thresh),np.sum(source_probs>0.5),
            "{:.2f}".format(np.mean(source_probs)),"{:.2f}".format(np.std(source_probs))])

    return stats


def run_em_on_window(n_cpgs, max_m, read_data, initial_high, initial_low):
    '''
    :param n_cpgs: N CpGs in window
    :param max_m: maximal coverage (N reads)
    :param read_data: np.array of methylation states
    :param initial_high: state A methylation inital values
    :param initial_low: state B methylation inital values
    :return: window BIC score, read labels, state thethas and
    posterior probability for state A
    '''
    em = two_epistate_EM(read_data, initial_high, initial_low, 0.5, 0.1, 10, use_theta_convergence=True)
    em.run_em()
    double_ll = em.get_ll()
    single_ll = max_single_epistate_ll(read_data)
    BIC = int(get_BIC(n_cpgs, max_m, single_ll, double_ll))
    thetas = em.get_thetas()
    probs = em.get_posterior_probability()
    return BIC, thetas, probs


def find_haplotypes(snp_matrix):
    '''
    find unique haplotypes in snp array
    :param snp_matrix: np array (dense) of snp data
    :return: unique haplotypes and indices
    '''
    haplotypes, indices = np.unique(snp_matrix, axis = 0, return_inverse = True)
    return haplotypes, indices

