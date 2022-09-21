###################################################
#
# Script: write_output.py
# Author: Irene Unterman
# Description: output EM results
###################################################
import sys
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools")

import os
import numpy as np
from epiread_tools.naming_conventions import *
from epiread_tools.em_utils import jsonconverter
import gzip
import json

def write_output(config, chrom, em_results, sample_stats):
    '''
    write all outputs to outdir
    :param chrom: chromosome for intervals
    :param em_results: em results (dict)
    :param sample_stats: array of stats per sample
    :param snp_stats: dict of haplotype data
    :param mapper:  mapping from rel to abs coordinates
    :param outdir: path for output files
    :return:
    '''
    write_sample_ids(name, mapper.sample_to_id, outdir)
    window_summary(name, chrom, em_results["windows"], em_results["BIC"], snp_stats.get("SNP_coords", {}),
                   mapper.ind_to_abs, mapper.snp_rel_to_abs, outdir)
    epistate_summary(name, chrom, em_results["windows"], em_results["Theta_A"],
                     em_results["Theta_B"], mapper.ind_to_abs, outdir)
    write_all_sample_summaries(name, chrom, mapper.get_sample_id_list(),em_results["windows"],
                               sample_stats, mapper.ind_to_abs, outdir)
    if snp_stats:
        write_all_haplotype_summaries(name, chrom, mapper.get_sample_id_list(),em_results["windows"],
                                      snp_stats, mapper.ind_to_abs, outdir)

def write_empty_output(name, outdir, snp_file=True, report_file=True):
    '''
    just create the output files without anything in them
    :param name: name for output files
    :param outdir: path for output files
    :param snp_file: create haplotype summary file
    :return:
    '''
    open(os.path.join(outdir, str(name)+"_sample_IDs.txt"), "a").close()
    open(os.path.join(outdir, str(name)+"_window_summary.bedgraph.gz"), "a").close()
    open(os.path.join(outdir, str(name)+"_epistate_summary.bed.gz"), "a").close()
    open(os.path.join(outdir, str(name)+"_sample_summary.bed.gz"), "a").close()
    if snp_file:
        open(os.path.join(outdir, str(name)+"_haplotype_summary.bed.gz"), "a").close()
    if report_file:
        open(os.path.join(outdir, str(name)+"_step_2.json"), "a").close()


def write_minimal_output(name, minimal_output, outdir):
    '''
    output minimal step_1 data for thresholding
    :param name:  name for output files
    :param minimal_output: minimal EM output as np array
    :param outdir:  path for output file
    :return:
    '''
    with gzip.open(os.path.join(outdir, str(name)+"_step_1.csv.gz"), "a") as outfile:
        np.savetxt(outfile, minimal_output, delimiter=TAB, fmt='%s')

def write_report_stats_to_json(name, report_stats, outdir):
    '''
    output dict with stats for html report
    :param name: name for output files
    :param report_stats: dict to output
    :param outdir: path for output file
    :return:
    '''
    with open(os.path.join(outdir, str(name)+"_step_2.json"), "a") as outfile:
        json.dump(report_stats, outfile, default=jsonconverter)

def write_sample_ids(name, sample_to_id, outdir):
    '''
    outout Sample   ID  per sample
    :param name: name for output files
    :param sample_to_id: dict from sample name to id
    :param outdir: path for output file
    :return:
    '''
    with open(os.path.join(outdir, str(name)+"_sample_IDs.txt"), "w") as outfile:
        for sample, id in sorted(sample_to_id.items(), key=lambda x: x[1]):
            outfile.write(sample+TAB+str(id)+"\n")

def window_summary(name, chrom, win_list, bic, snp_locs, ind_to_abs, snp_rel_to_abs, outdir):
    '''
    one line per window:
    chr start end BIC CpG_rel_pos SNP_rel_pos SNP_alt_alleles
    chr2 100 125 -1250 0,20,34,88,125 56,61    A,T
    :param name: name for output files
    :param chrom: chromosome for intervals
    :param win_list: list of relative coordinates of windows
    :param bic: list of BIC values per window
    :param snp_locs: list of SNP location per window
    :param ind_to_abs: mapping from matrix index coordinates to absolute
    :param snp_rel_to_abs:  mapping from relative coordinates to absolute for snp data
    :param outdir: path for output file
    :return:
    '''

    output_array = np.zeros(shape=(len(win_list), 6), dtype=object)
    for i, (start_ind, end_ind), in enumerate(win_list):
        win_snps = snp_locs.get(i, np.array([]))
        cpg_positions = np.array([ind_to_abs(i) - ind_to_abs(start_ind) for i in range(start_ind, end_ind)])
        snp_positions = np.array([snp_rel_to_abs(snp) - ind_to_abs(start_ind) for snp in win_snps])
        output_array[i,:] = chrom, ind_to_abs(start_ind), \
                            ind_to_abs(end_ind - 1) + 1, bic[i], format_array(cpg_positions), format_array(snp_positions)
    with gzip.open(os.path.join(outdir, str(name)+"_window_summary.bedgraph.gz"), "a") as outfile:
        np.savetxt(outfile, output_array, delimiter=TAB, fmt='%s')


def epistate_summary(name, chrom, win_list, theta_a, theta_b, ind_to_abs, outdir):
    '''
    two lines per window (one per epistate)
    chr start end state theta    alt_allele_freq
    chr2 100  125  A     0.8,0.7,0.85   0.4,0.1
    :param name: name for output files
    :param chrom: chromosome for intervals
    :param win_list: list of relative coordinates of windows
    :param theta_a: list of theta for state A per window
    :param theta_b: list of theta for state b per window
    :param ind_to_abs: dict from matrix index coordinates to genomic
    :param outdir: path for output file
    :return:
    '''

    output_array = np.zeros((len(theta_a)*2, 5), dtype=object)
    for i, (start_ind, end_ind) in enumerate(win_list):
        output_array[i,:] = chrom, ind_to_abs(start_ind), \
                            ind_to_abs(end_ind - 1) + 1, "A", format_array(theta_a[i])
    for i, (start_ind, end_ind) in enumerate(win_list):
        output_array[i+len(theta_a),:] = chrom, ind_to_abs(start_ind), \
                                         ind_to_abs(end_ind - 1) + 1, "B", format_array(theta_b[i])
    with gzip.open(os.path.join(outdir, str(name)+"_epistate_summary.bed.gz"), "a") as outfile:
        np.savetxt(outfile, output_array, delimiter=TAB, fmt='%s')

def format_array(array):
    '''
    :param array: np array
    :return: array formatted for output
    '''
    return np.array2string(array, max_line_width=100000,separator=",", precision=3)

def format_haplotype(haplotype):
    '''
    format haplotype for output
    :param haplotype: np array of SNPs, e.g. [1,0,2]
    :return: formatted string, e.g. "A.C"
    '''
    return "".join([ind_to_dna[x] for x in haplotype])


def write_all_sample_summaries(name, chrom, sample_ids, window_list, output, ind_to_abs, outdir):
    '''
    write one record per sample per window
    :param name: name for output files
    :param chrom: chromosome for intervals
    :param sample_ids: list of all sample IDs
    :param window_list: list of relative coordinates of windows
    :param output: sample summaries to write
    :param ind_to_abs: dict from matrix index coordinates to genomic
    :param outdir: path for output file
    :return:
    '''
    n_cols = 10 #number of output stats
    out_arrays = []
    for sample_id in sample_ids:
        has_data_filter = np.any(output[sample_id], axis=1)
        sample_data = output[sample_id][has_data_filter, :]
        sample_windows = np.array(window_list)[has_data_filter]
        out_arrays.append(sample_summary(chrom, sample_id, sample_windows, sample_data, ind_to_abs, n_cols))
    with gzip.open(os.path.join(outdir, str(name)+"_sample_summary.bed.gz"), "a") as outfile:
        np.savetxt(outfile, np.vstack(tuple(out_arrays)), delimiter=TAB, fmt='%s')

def sample_summary(chrom, sample_id, window_list, sample_stats, ind_to_abs, n_cols):
    '''
    chr start end sample n_read n>0.9, n<0.1, n>0.5, pp_mean, pp_stdev, pps(optional)
    chr2 100  125  0     54      18     20      19      0.2      0.01    0.8,0.1,0.4
    :param chrom: chromosome for intervals
    :param sample_id: id of sample
    :param sample_stats: output of specific sample
    :param ind_to_abs: dict from relative coordinates to genomic
    :param n_cols: number of columns in output (for formatting)
    :return:np array of records per window with sample
    '''
    output_array = np.zeros((len(window_list), n_cols), dtype=object)
    for i, (start_ind, end_ind) in enumerate(window_list):
        output_array[i] = chrom, ind_to_abs(start_ind), \
                          ind_to_abs(end_ind - 1) + 1, sample_id, *sample_stats[i]
    return output_array

def write_all_haplotype_summaries(name, chrom, sample_ids, window_list, snp_output, ind_to_abs, outdir):
    '''
    write one record per sample per window
    :param name: name for output files
    :param chrom: chromosome for intervals
    :param sample_ids: list of all sample IDs
    :param window_list: list of matrix index coordinates of windows
    :param snp_output: haplotype results (dict)
    :param ind_to_abs: dict from matrix index coordinates to genomic
    :param outdir: path for output file
    :return:
    '''
    n_cols = 11 #number of output stats
    out_arrays = []
    for sample_id in sample_ids:
        if sample_id in snp_output:
            out_arrays.append(haplotype_summary(chrom, sample_id, window_list,
                                                snp_output[sample_id], ind_to_abs, n_cols))
    if out_arrays:
        merged_outputs = np.vstack(tuple(out_arrays))
    else:
        merged_outputs = np.array([])
    with gzip.open(os.path.join(outdir, str(name)+"_haplotype_summary.bed.gz"), "a") as outfile:
        np.savetxt(outfile, merged_outputs, delimiter=TAB, fmt='%s')

def haplotype_summary(chrom, sample_id, window_list, sample_stats, ind_to_abs, n_cols):
    '''
    chr start end sample haplotype n_read n>0.9, n<0.1, n>0.5, pp_mean, pp_stdev, pps(optional)
    chr2 100  125  0    AAT         54      18     20      19      0.2      0.01    0.8,0.1,0.4
    :param chrom: chromosome for intervals
    :param sample_id: source id
    :param window_list: list of matrix index coordinates of windows
    :param sample_stats: dict of stats to output
    :param ind_to_abs: dict from matrix index coordinates to genomic
    :param n_cols: number of columns in output
    :return:
    '''
    sample_haplotypes = []
    for win_index, hap_array in sample_stats.items():
        start_ind, end_ind = window_list[win_index]
        formatted_array = np.zeros((hap_array.shape[0], n_cols), dtype=object)
        formatted_array[:,0:3] = chrom, ind_to_abs(start_ind), ind_to_abs(end_ind - 1) + 1
        formatted_array[:,3] = sample_id
        formatted_array[:,4] = [format_haplotype(hap) for hap in hap_array[:,0]]
        formatted_array[:,5:] = hap_array[:,1:]
        sample_haplotypes.append(formatted_array)
    return np.vstack(sample_haplotypes)


