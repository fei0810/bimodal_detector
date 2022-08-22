import sys
import os
import numpy as np
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools")


from epiread_tools.epiparser import parse_epireads
from write_output import write_output, write_minimal_output, write_report_stats_to_json, write_empty_output
from run_em import run_em, make_window_list, get_all_stats, get_all_snp_stats
from epiread_tools.em_utils import GenomicInterval, Mapper
from epiread_tools.naming_conventions import *
from filter_bic import *
import pandas as pd
import scipy.sparse as sp
import json
import logging

#%%
class Runner:

    def __init__(self, genomic_intervals, cpg_locations,
                 epiread_files, parse_snps=True, walk_on_list=True, win_size=5, step_size=1, get_pp=False,
                 name=None,one_based=True, outdir=None, logfile="log.log"):
        '''
        runs EM start to finish
        :param genomic_intervals: list of intervals in chrN:0-100 format.
        :param cpg_locations: File with CpG coordinates in bed format (chrom start end), tab delimited
        :param epiread_files: list of epiread file names (paths)
        :param parse_snps: Output SNP data
        :param walk_on_list: Create a genome walker. If false, each genomic interval will be processed
        as one window
        :param win_size: If a genome walker is required, number of CpGs per window
        :param step_size: If a genome walker is required, step between windows
        :param get_pp: Output posterior probabilities  for state A per read
        :param name: run name for output files
        :param outdir: directory for output files
        '''
        self.original_intervals = genomic_intervals
        self.genomic_intervals = self.init_genomic_intervals(genomic_intervals)
        self.cpg_locations = cpg_locations
        self.epiread_files = epiread_files
        self.parse_snps = parse_snps
        self.walk_on_list = walk_on_list
        self.win_size = win_size
        self.step_size = step_size
        self.get_pp = get_pp
        self.one_based = one_based
        self.name = name
        self.outdir = outdir
        if not self.name: #no name defined
            self.name = self.default_name()
        if not self.outdir: #no output directory, save in running directory
            self.outdir = os.getcwd()
            fh = logging.FileHandler(self.logfile)
            fh.setLevel(logging.DEBUG)
        self.logfile = logfile
        self.logger = self.init_logger(self.logfile)
        #log init parameters
        self.logger.debug(str(self))

    def init_logger(self, logfile, level=logging.DEBUG):
        logger = logging.getLogger(self.name)
        logger.setLevel(level)
        fh = logging.FileHandler(logfile)
        fh.setLevel(level)
        logger.addHandler(fh)
        return logger

    def __repr__(self):
        return """Name: %s, CpG location file : %s, parse SNPs: %s, walk on list: %s,
                win_size: %d, step size %d, one based %s""" % (self.name, self.cpg_locations,
            str(self.parse_snps), str(self.walk_on_list), self.win_size, self.step_size, str(self.one_based))

    def init_genomic_intervals(self, genomic_intervals):
        '''
        parse list of intervals to GenomicInterval objects
        :param genomic_intervals: list of intervals in chrN:0-100 format
        :return: list of GenomicInterval objects
        '''
        return [GenomicInterval(interval) for interval in genomic_intervals]

    def default_name(self):
        '''
        if no name is supplied, provide range of intervals
        :return: str of genomic intervals start-end range
        '''
        return self.genomic_intervals[0].chrom + ":" + str(self.genomic_intervals[0].start) +\
        "-"+ self.genomic_intervals[-1].chrom + ":" +  str(self.genomic_intervals[-1].end)

    def get_raw_reads(self, intervals, chrom):
        '''
        retrieve parsed raw reads from intervals on one chromosome
        :param intervals: list of GenomicInterval objects
        :param chrom: "chrN"
        :return:
        '''
        intervals = sorted(intervals, key=lambda x: x.start)
        methylation_matrix, snp_matrix, mapper = parse_epireads(chrom, intervals, self.epiread_files,
                                                                self.cpg_locations, self.one_based, self.parse_snps)
        nrows, ncols = methylation_matrix.shape
        row_names = [mapper.index_to_source(i) for i in range(nrows)]
        col_names = [mapper.ind_to_abs(i) for i in range(ncols)]
        res = pd.DataFrame(methylation_matrix.toarray(), columns=col_names)
        res["source"] = row_names
        res.to_csv(os.path.join(self.outdir, str(self.name)+"_raw_reads.csv"), index=False)


    def run_one_chromosome(self, intervals, chrom):
        '''
        run em on intervals from one chromosome
        :param intervals: list of GenomicInterval objects
        :param chrom: "chrN"
        :return:
        '''
        #make sure intervals are sorted
        intervals = sorted(intervals, key=lambda x: x.start)
        methylation_matrix, snp_matrix, mapper = parse_epireads(chrom, intervals, self.epiread_files,
                                                                self.cpg_locations, self.one_based, self.parse_snps)
        window_list = make_window_list(mapper.get_ind_intervals(intervals), self.walk_on_list, self.win_size, self.step_size)
        em_results = run_em(methylation_matrix, window_list)
        stats = get_all_stats(em_results["Indices"], em_results["Probs"], mapper.index_to_source, mapper.get_sample_id_list(),
                            self.get_pp)
        if self.parse_snps:
            snp_stats = get_all_snp_stats(snp_matrix, em_results["Indices"], em_results["Probs"], mapper.index_to_source,
                                          self.get_pp)
        else:
            snp_stats = {}
        write_output(self.name, chrom, em_results, stats, snp_stats, mapper, self.outdir)

    def split_intervals_to_chromosomes(self):
        '''
        find all chromosomes in intervals
        :return:
        '''
        by_chrom = defaultdict(list)
        for interval in self.genomic_intervals:
            by_chrom[interval.chrom].append(interval)
        self.intervals_per_chrom = by_chrom

    def run_multiple_chromosomes(self):
        '''
        run each chromosome separately
        :return:
        '''
        self.split_intervals_to_chromosomes()
        for chrom, intervals in self.intervals_per_chrom.items():
            self.run_one_chromosome(intervals, chrom)

    def run_parameter_estimation(self, intervals, chrom):
        intervals = sorted(intervals, key=lambda x: x.start)
        methylation_matrix, snp_matrix, mapper = parse_epireads(chrom, intervals,
                                                                self.epiread_files, self.cpg_locations,self.one_based, self.parse_snps)
        window_list = make_window_list(mapper.get_ind_intervals(intervals), self.walk_on_list, self.win_size, self.step_size)
        em_results = run_em(methylation_matrix, window_list)
        stats = get_all_stats(em_results["Indices"], em_results["Probs"],
                              mapper.index_to_source, mapper.get_sample_id_list(), False)
        pp_vectors = get_mean_pp_vec_from_stats(stats)
        win_starts = np.array([mapper.ind_to_rel(x[0]) for x in em_results["windows"]]).reshape((-1,1))
        win_ends = np.array([mapper.ind_to_rel(x[1]-1) + 1 for x in em_results["windows"]]).reshape((-1,1))
        write_minimal_output(self.name, np.hstack([win_starts, win_ends, em_results["BIC"].reshape((-1,1)), pp_vectors]),
                             self.outdir)


    def two_step_run(self, intervals, chrom, loose_threshold, strict_threshold, similarity_threshold,
                     minimal_improvement=1.1):
        '''
        run two_step em for one chromosome
        :param intervals: list of GenomicInterval objects
        :param chrom: "chrN"
        :param loose_threshold: first BIC thresh
        :param strict_threshold: second BIC thresh
        :param similarity_threshold: thresh for merging
        :return:
        '''
        child_runner = TwoStepRunner(self.original_intervals, self.cpg_locations,
                 self.epiread_files, self.parse_snps, self.walk_on_list, self.win_size, self.step_size, self.get_pp,
                 self.name, self.one_based, self.outdir, self.logfile)
        child_runner.set_run_params(intervals, chrom, loose_threshold, strict_threshold, similarity_threshold,
                     minimal_improvement)
        child_runner.run_two_steps()


class TwoStepRunner(Runner):

    def __init__(self,genomic_intervals, cpg_locations,
                 epiread_files, parse_snps=True, walk_on_list=True, win_size=5, step_size=1, get_pp=False,
                 name=None,one_based=True, outdir=None, logfile="log.log"):
        super().__init__(genomic_intervals, cpg_locations,
                 epiread_files, parse_snps, walk_on_list, win_size, step_size, get_pp,
                 name,one_based, outdir, logfile)
        #init outputs
        self.em_results, self.step_two_em  = self.get_new_em_dict(), self.get_new_em_dict()
        self.old_window_em_results, self.new_window_em_results = self.get_new_em_dict(), self.get_new_em_dict()
        self.old_window_stats, self.stats_for_new = {}, {}
        self.old_window_snp_stats, self.snp_stats_for_new = {}, {}
        self.to_keep, self.to_rerun = [], []
        self.lowest_pre_merge_bics, self.post_merge_bics = np.array([]), np.array([])

    def set_run_params(self, intervals, chrom, loose_threshold, strict_threshold, similarity_threshold,
                     minimal_improvement=1.1, stat_report=True):
        '''
        run two_step em for one chromosome
        :param intervals: list of GenomicInterval objects
        :param chrom: "chrN"
        :param loose_threshold: first BIC thresh
        :param strict_threshold: second BIC thresh
        :param similarity_threshold: thresh for merging
        :return:
        '''
        self.intervals = sorted(intervals, key=lambda x: x.start)
        self.chrom = chrom
        self.loose_threshold = loose_threshold
        self.strict_threshold = strict_threshold
        self.similarity_threshold = similarity_threshold
        self.minimal_improvement = minimal_improvement
        self.stat_report = stat_report

    def run_two_steps(self):
        '''
        runs two step em according to params
        :return:
        '''
        self.parse_reads()
        if not self.methylation_matrix.nnz: #empty matrix
            self.logger.debug("input matrix was empty for chunk "+self.name)
            write_empty_output(self.name, self.outdir, self.parse_snps, self.stat_report)
            ### maybe write stats anyway?
            return
        self.update_window_list()

        self.em_results.update(run_em(self.methylation_matrix, self.window_list)) #step 1
        self.update_step_1_stats()

        self.first_filter()
        if not self.filtered_windows:
            write_empty_output(self.name, self.outdir, self.parse_snps, self.stat_report)
            #write report amd exit
            if self.stat_report:
                self.write_report()
            return
        self.merge_windows()

        self.step_two_em.update(run_em(self.methylation_matrix, self.rel_new_windows))  #step 2
        self.unmerge_windows()

        self.second_filter()
        self.update_step_2_stats()
        if self.parse_snps:
            self.update_snp_stats()
        self.main_output()
        if self.stat_report:
            self.write_report()


    def get_new_em_dict(self):
        '''
        init empty em results dict
        :return: em dict
        '''
        em_dict =  {'windows': [],
                     'BIC': [],
                     'Theta_A': [],
                     'Theta_B': [],
                     'Probs': [],
                     'Indices': []}
        return em_dict

    def parse_reads(self):
        '''
        set data amd mapper
        :return:
        '''
        self.methylation_matrix, self.snp_matrix, self.mapper = parse_epireads(self.chrom, self.intervals,
                                    self.epiread_files, self.cpg_locations,self.one_based, self.parse_snps)

    def update_window_list(self):
        '''
        set list of windows for em
        :return:
        '''
        self.window_list = make_window_list(self.mapper.get_ind_intervals(self.intervals),
                                            self.walk_on_list, self.win_size, self.step_size)
    def update_step_1_stats(self):
        '''
        set summary statistics matrix and retreive posterior probabilities
        :return:
        '''
        self.stats = get_all_stats(self.em_results["Indices"], self.em_results["Probs"],
                              self.mapper.index_to_source, self.mapper.get_sample_id_list(), self.get_pp)
        self.pp_vectors = get_mean_pp_vec_from_stats(self.stats)

    def first_filter(self):
        '''
        apply BIC filter to em results
        :return:
        '''
        self.first_filter = apply_threshold(self.em_results["BIC"], self.loose_threshold)
        self.filtered_windows = filter_list(self.em_results["windows"], self.first_filter)

    def merge_windows(self):
        '''
        merge overlapping windows
        :return:
        '''
        #where to merge
        self.divs = merge_indices(self.filtered_windows, self.pp_vectors[self.first_filter], self.similarity_threshold)
        self.to_keep, self.to_rerun = split_old_and_new_windows(self.divs)
        self.rel_new_windows = merge_windows(self.filtered_windows, self.to_rerun)
        self.lowest_pre_merge_bics = get_bic_scores_pre_merging(filter_list(self.em_results["BIC"],self.first_filter),
                                                                self.to_rerun)

    def unmerge_windows(self):
        '''
        if minimal improvement hasn't been met,
        unmerge windows
        :return:
        '''
        #new_windows

        self.post_merge_bics = self.step_two_em["BIC"]
        self.extra_indices_to_keep, self.filter_for_merged_windows = unmerging(self.lowest_pre_merge_bics, self.post_merge_bics,
                                                                     self.minimal_improvement, self.to_rerun)
        self.to_keep += self.extra_indices_to_keep #keep also unmerged windows
        self.step_two_em = filter_em_results(self.step_two_em, self.filter_for_merged_windows) #remove unmerged from step 2

    def second_filter(self):
        '''
        apply strict filter to all windows
        :return:
        '''
        #old windows:
        #bool filter
        self.strict_for_old = apply_threshold(self.em_results["BIC"][self.first_filter][self.to_keep], self.strict_threshold)
        #window indices
        self.old_windows_post_strict = np.arange(len(self.em_results["windows"]))[self.first_filter][self.to_keep][self.strict_for_old]
        self.old_window_em_results = filter_em_results(self.em_results, self.old_windows_post_strict, indices=True)

        self.strict_for_new = apply_threshold(np.array(self.step_two_em["BIC"]),self.strict_threshold)
        self.new_window_em_results = filter_em_results(self.step_two_em, self.strict_for_new)


    def update_step_2_stats(self):
        '''
        find stats for old and new windows
        :return:
        '''
        self.old_window_stats = self.stats[:,self.old_windows_post_strict,:]
        self.stats_for_new = get_all_stats(self.new_window_em_results["Indices"],self.new_window_em_results["Probs"],
                        self.mapper.index_to_source, self.mapper.get_sample_id_list(),self.get_pp)


    def update_snp_stats(self):
        '''
        find snp stats for old and new windows
        :return:
        '''
        self.old_window_snp_stats = get_all_snp_stats(self.snp_matrix, self.old_window_em_results["Indices"],
                                                 self.old_window_em_results["Probs"],
                                                 self.mapper.index_to_source, self.get_pp)
        self.snp_stats_for_new = get_all_snp_stats(self.snp_matrix, self.new_window_em_results["Indices"], self.new_window_em_results["Probs"],
                                              self.mapper.index_to_source, self.get_pp)

    def main_output(self):
        '''
        write output for old and new windows
        :return:
        '''
        write_output(self.name, self.chrom, self.old_window_em_results, self.old_window_stats,
                     self.old_window_snp_stats, self.mapper, self.outdir)
        write_output(self.name, self.chrom, self.new_window_em_results, self.stats_for_new,
                     self.snp_stats_for_new, self.mapper, self.outdir)

    def write_report(self):
        report_stats = {
            #counts
                        "n_windows": len(self.em_results["BIC"]), "n_post_loose": len(self.filtered_windows),
                        "n_merged": len(self.step_two_em["windows"]),
                        "n_post_strict_merged": len(self.new_window_em_results["BIC"]),
                        "n_post_strict_unmerged" : len(self.old_window_em_results["BIC"]),
                        "n_post_loose_unmerged": len(self.to_keep),
            #value distributions
                        "original_bics": np.sort(self.em_results["BIC"]), "lowest_pre_merge_bics": self.lowest_pre_merge_bics,
                        "post_merge_bics": self.post_merge_bics,
                        "post_strict_bic": np.hstack([self.old_window_em_results["BIC"], self.new_window_em_results["BIC"]]),
                        "win_sizes": [x[1]-x[0] for x in self.old_window_em_results["windows"]] +
                                     [x[1]-x[0] for x in self.new_window_em_results["windows"]]}
        write_report_stats_to_json(self.name, report_stats, self.outdir)



#%%
#TODO: add option to save reads with pp, in bubble_plot format
if __name__=="__main__":

    epiread_files = ["/Users/ireneu/PycharmProjects/proj-epireads/runs/Panc_GRAIL_UPenn/chunk_941.epiread.gz"]
    with open("/Users/ireneu/PycharmProjects/proj-epireads/runs/Panc_GRAIL_UPenn/whole_genome_chunks.json") as infile:
        chunks = json.load(infile)
    genomic_intervals = chunks[941][:7]
    cpg_locations = "/Users/ireneu/PycharmProjects/proj-epireads/Whole_genome/sample_inputs/hg19.CpG.bed.gz"
    runner = Runner(genomic_intervals, cpg_locations,epiread_files, name="problem_run", outdir="/Users/ireneu/PycharmProjects/proj-epireads/runs/Panc_GRAIL_UPenn/",
                    one_based=True, parse_snps=True)
    runner.two_step_run(runner.genomic_intervals, "chr19", -551, -551, 0.34)

