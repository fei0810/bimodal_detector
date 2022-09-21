import sys
import os
import numpy as np
sys.path.append("/Users/ireneu/PycharmProjects/bimodal_detector/bimodal_detector") ###
sys.path.append("/Users/ireneu/PycharmProjects/bimodal_detector/") ###
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools/") ###

from epiread_tools.epiparser import EpireadReader, CoordsEpiread, epiformat_to_reader
from epiread_tools.naming_conventions import *
from epiread_tools.em_utils import calc_coverage, calc_methylated
from write_output import write_output, write_minimal_output, write_report_stats_to_json, write_empty_output
from run_em import run_em, get_all_stats, get_all_snp_stats, do_walk_on_list
from filter_bic import *
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
        self.reader = epiformat_to_reader[self.config["epiformat"]]
        self.outdir = self.config["outdir"] if len(self.config["outdir"]) else os.getcwd()
        if self.config["verbose"]:
            self.init_logger()

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

    def run(self):
        reader = self.reader(self.config)
        self.interval_order, self.matrices, self.cpgs = reader.get_matrices_for_intervals()
        self.sources = reader.sources
        for i, interval in enumerate(self.interval_order): #interval should never span more than 1 chromosome
            window_list = [(0, self.matrices[i].shape[1])]
            if self.config["walk_on_list"]:
                window_list = do_walk_on_list(window_list, self.config["window_size"], self.config["step_size"])
            em_results = run_em(self.matrices[i], window_list)
            stats = get_all_stats(em_results["Indices"], em_results["Probs"], dict(zip(np.arange(len(self.sources[i])), self.sources[i])),
                                  len(self.config["epiread_files"]), self.config["get_pp"])
            write_output(self.config, interval.chrom, em_results, stats)

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

# if __name__ == '__main__':
#     main()

config = {"genomic_intervals": ["chr1:750000-755000","chr2:750000-755000"],
  "cpg_coordinates": "/Users/ireneu/PycharmProjects/old_in-silico_deconvolution/debugging/hg19.CpG.bed.sorted.gz",
  "epiread_files": ["/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sample_epiread_with_A.epiread.gz"],
  "outdir": "output",
  "epiformat": "old_epiread_A",
  "header": False,
  "bedfile": False,
  "parse_snps": False,
    "get_pp":False,
  "walk_on_list": True,
    "verbose" : False,
  "window_size": 5,
  "step_size": 1,
    "name": "banana",
  "logfile": "log.log"}
runner = Runner(config)
runner.run()

#TODO:
# rewrite output
# handle SNPS?
# add get PP option
# implement two-step
# implement param estimation
# add logging
# add tests