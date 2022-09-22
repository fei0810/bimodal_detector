import sys
import os
import numpy as np
sys.path.append("/Users/ireneu/PycharmProjects/bimodal_detector/") ###
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools/") ###

from epiread_tools.epiparser import EpireadReader, CoordsEpiread, epiformat_to_reader
from epiread_tools.naming_conventions import *
from epiread_tools.em_utils import calc_coverage, calc_methylated
from bimodal_detector.run_em import run_em, get_all_stats, get_all_snp_stats, do_walk_on_list
from bimodal_detector.filter_bic import *
from bimodal_detector.output_utils import relative_intervals_to_abs, cpg_positions_in_interval, format_array
import pandas as pd
import scipy.sparse as sp
import json
import logging
import click

class Runner:

    def __init__(self, config):
        '''
        runs EM start to finish
        :param config: dict with run params
        '''
        self.config = config
        self.name = self.config["name"]
        self.reader = epiformat_to_reader[self.config["epiformat"]]
        self.outdir = self.config["outdir"] if len(self.config["outdir"]) else os.getcwd()
        if self.config["verbose"]:
            self.init_logger()
        self.results, self.stats = [], []

    def init_logger(self, level=logging.DEBUG):
        logger = logging.getLogger(self.name)
        logger.setLevel(level)
        fh = logging.FileHandler(self.config["logfile"])
        fh.setLevel(level)
        logger.addHandler(fh)
        return logger

    def __repr__(self):
        return """Name: %s, CpG location file : %s, parse SNPs: %s, walk on list: %s,
                win_size: %d, step size %d""" % (self.name, self.config["cpg_coordinates"],
            str(self.config["parse_snps"]), str(self.config["walk_on_list"]),
                                self.config["window_size"], self.config["step_size"])

    def read(self):
        reader = self.reader(self.config)
        self.interval_order, self.matrices, self.cpgs = reader.get_matrices_for_intervals()
        self.sources = reader.get_sources()

    def em_all(self):
        for i, interval in enumerate(self.interval_order): #interval should never span more than 1 chromosome
            window_list = [(0, self.matrices[i].shape[1])]
            if self.config["walk_on_list"]:
                window_list = list(do_walk_on_list(window_list, self.config["window_size"], self.config["step_size"]))
            em_results = run_em(self.matrices[i], window_list)
            stats = get_all_stats(em_results["Indices"], em_results["Probs"], dict(zip(np.arange(len(self.sources[i])), self.sources[i])),
                                  len(self.config["epiread_files"]), self.config["get_pp"])
            self.results.append(em_results)
            self.stats.append(stats)

    def write_sample_ids(self):
        '''
        outout Sample   ID  per sample
        :return:
        '''
        samples = ["ALL"] + self.config["epiread_files"]
        ids = np.arange(len(samples))
        with open(os.path.join(self.outdir, str(self.name) + "_sample_IDs.txt"), "w") as outfile:
            for sample, id in zip(samples, ids):
                outfile.write(sample + TAB + str(id) + "\n")

    def write_window_summary(self): #TODO: add snps
        '''
            one line per window:
        chr start end BIC CpG_rel_pos SNP_rel_pos SNP_alt_alleles
        chr2 100 125 -1250 0,20,34,88,125 56,61    A,T
        :param self:
        :return:
        '''
        abs_windows = []
        rel_positions = []
        for i, interval in enumerate(self.interval_order):
            abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], self.results[i]["windows"]))
            for x in cpg_positions_in_interval(self.cpgs[i], self.results[i]["windows"]):
                rel_positions.append(format_array(x))
        output_array = np.hstack([np.vstack(abs_windows),
                        np.hstack([ self.results[x]["BIC"] for x in range(len(self.interval_order))]).reshape(-1,1),
                        np.vstack(rel_positions)])
        with gzip.open(os.path.join(self.outdir, str(self.name) + "_window_summary.bedgraph.gz"), "a") as outfile:
            np.savetxt(outfile, output_array, delimiter=TAB, fmt='%s')

    def write_sample_summary(self): #TODO
        '''
        chr start end sample n_read n>0.9, n<0.1, n>0.5, pp_mean, pp_stdev, pps(optional)
        chr2 100  125  0     54      18     20      19      0.2      0.01    0.8,0.1,0.4
        :return:
        '''
        abs_windows = []
        for i, interval in enumerate(self.interval_order):
            abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], self.results[i]["windows"]))
        a = np.vstack(abs_windows)
        b = np.hstack(self.stats)
        output_array = np.vstack([np.hstack([a, np.full(a.shape[0], x).reshape(-1,1), b[x,:,:]]) for x in range(b.shape[0])])
        with gzip.open(os.path.join(self.outdir, str(self.name) + "_sample_summary.bed.gz"), "a") as outfile:
            np.savetxt(outfile, output_array, delimiter=TAB, fmt='%s')

    def run(self):
        self.read()
        self.em_all()
        self.write_sample_ids()
        self.write_window_summary()
        self.write_sample_summary()


class ParamEstimator(Runner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_parameter_estimation(self, intervals, chrom):
        pass
        # intervals = sorted(intervals, key=lambda x: x.start)
        # methylation_matrix, snp_matrix, mapper = parse_epireads(chrom, intervals,
        #                                                         self.epiread_files, self.cpg_locations,self.one_based, self.parse_snps)
        # window_list = make_window_list(mapper.get_ind_intervals(intervals), self.walk_on_list, self.win_size, self.step_size)
        # em_results = run_em(methylation_matrix, window_list)
        # stats = get_all_stats(em_results["Indices"], em_results["Probs"],
        #                       mapper.index_to_source, mapper.get_sample_id_list(), False)
        # pp_vectors = get_mean_pp_vec_from_stats(stats)
        # win_starts = np.array([mapper.ind_to_rel(x[0]) for x in em_results["windows"]]).reshape((-1,1))
        # win_ends = np.array([mapper.ind_to_rel(x[1]-1) + 1 for x in em_results["windows"]]).reshape((-1,1))
        # write_minimal_output(self.name, np.hstack([win_starts, win_ends, em_results["BIC"].reshape((-1,1)), pp_vectors]),
        #                      self.outdir)

    def write_minimal_output(self):
        pass

    def run(self):
        self.read()
        self.em_all()
        self.write_minimal_output()

class TwoStepRunner(Runner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('-j', '--json', help='run from json config file')
@click.version_option()
@click.pass_context
def main(ctx, **kwargs):
    """deconvolute epiread file using atlas"""
    with open(kwargs["json"], "r") as jconfig:
        config = json.load(jconfig)
    config.update(kwargs)
    config.update(dict([item.strip('--').split('=') for item in ctx.args]))

    if config["run_type"]=='basic':
        runner=Runner
    elif config["run_type"]=='param_estimation':
        runner=ParamEstimator
    elif config["run_type"]=='two-step':
        runner=TwoStepRunner


    em_runner = runner(config)
    em_runner.run()

if __name__ == '__main__':
    main()

# config = {"genomic_intervals": ["chr1:750000-755000","chr2:750000-755000"],
#   "cpg_coordinates": "/Users/ireneu/PycharmProjects/old_in-silico_deconvolution/debugging/hg19.CpG.bed.sorted.gz",
#   "epiread_files": ["/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sample_epiread_with_A.epiread.gz"],
#   "outdir": "output",
#   "epiformat": "old_epiread_A",
#   "header": False,
#   "bedfile": False,
#   "parse_snps": False,
#     "get_pp":False,
#   "walk_on_list": True,
#     "verbose" : False,
#   "window_size": 5,
#   "step_size": 1,
#     "name": "banana",
#   "logfile": "log.log"}
# runner = Runner(config)
# runner.run()

#TODO:
# handle SNPS?
# add get PP option
# implement two-step
# implement param estimation
# add logging
# add tests