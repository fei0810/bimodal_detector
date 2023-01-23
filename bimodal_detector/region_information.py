'''
find how informative each region is for each cell type
per model
'''
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools/") ###
sys.path.append("/Users/ireneu/PycharmProjects/deconvolution_models/deconvolution_models") ###
from bimodal_detector.runner_utils import relative_intervals_to_abs
from epiread_tools.naming_conventions import *
from scipy.special import logsumexp
from bimodal_detector.runners import AtlasEstimator
from bimodal_detector.run_em import run_em, get_all_stats
from collections import defaultdict
from bimodal_detector.run_em import do_walk_on_list, clean_section
import os
import scipy as sp
from epiread_tools.em_utils import GenomicInterval
from itertools import compress


class InfoRunner(AtlasEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_thetas(self):
        self.theta_A = []
        self.theta_B = []
        for i, interval in enumerate(self.interval_order):
            if "windows" in self.results[i] and self.results[i]["windows"]:  # any windows with results
                thetas = zip(self.results[i]["Theta_A"], self.results[i]["Theta_B"])
                self.theta_A.append([])
                self.theta_B.append([])
                for m, n in thetas:
                   self.theta_A[i].append(m)
                   self.theta_B[i].append(n)
            else:
                self.theta_A.append([])
                self.theta_B.append([])


    def calc_info(self, model):
        res = defaultdict(list)
        self.input_windows = []
        self.abs_windows = []
        betas=[]
        for i, interval in enumerate(self.interval_order):
            if type(self.matrices[i]) is list and not self.matrices[i].any():
                continue
            window_list = [(0, self.matrices[i].shape[1])]
            if self.config["walk_on_list"]:
                window_list = list(do_walk_on_list(window_list, self.config["window_size"], self.config["step_size"]))
            methylation_matrix = self.matrices[i].tocsc()

            for j, (start, stop) in enumerate(window_list):
                section, sec_read_ind = clean_section(methylation_matrix, start, stop)
                if np.sum(section) == 0: #no data
                    continue
                self.abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], [(start, stop)])[0])
                self.input_windows.append((interval.chrom, interval.start, interval.end))
                high, low = self.theta_A[i][j], self.theta_B[i][j]
                lt = self.lambdas[i][:,j].flatten()
                src = self.sources[i][sec_read_ind]
                source_labels = np.array(self.labels)[src - 1]  # adjusted for index
                beta = np.vstack([calc_beta(section[np.array(source_labels) == t, :]) for t in self.config["cell_types"]])
                betas.append(self.cell_types[np.argmin(np.nanmean(beta, axis=1))])
                for k, cell in enumerate(self.config["cell_types"]):
                    reads = section[np.array(source_labels) == cell, :]
                    if model == "epistate-plus":
                        res[cell].append(epistate_plus_info(lt, high, low, reads)[k])
                    else:
                        res[cell].append(model_to_fun[model](beta, reads)[k])
        a = pd.DataFrame(self.input_windows, columns=["input_chrom", "input_start", "input_end"])
        b = pd.DataFrame(self.abs_windows, columns=["window_chrom", "window_start", "window_end"])
        c = pd.DataFrame({"lowest_beta": betas})
        d = pd.DataFrame(res, columns= self.cell_types)

        return pd.concat([a,b,c,d], axis=1)

    def region_stats(self):
        input_windows = []
        abs_windows = []
        bic = []
        med_cpg = []
        percent_single = []
        for i, interval in enumerate(self.interval_order):
            if "windows" in self.results[i] and self.results[i]["windows"]:  # any windows with results
                abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], self.results[i]["windows"]))
                input_windows.append([(interval.chrom, interval.start, interval.end)]*len(self.results[i]["windows"]))
                bic.extend(self.results[i]["BIC"])
                med_cpg.extend(self.results[i]["median_cpg"])
                percent_single.extend(self.results[i]["percent_single"])
        a = pd.DataFrame(self.input_windows, columns=["input_chrom", "input_start", "input_end"])
        b = pd.DataFrame(self.abs_windows, columns=["window_chrom", "window_start", "window_end"])
        c = pd.DataFrame({"BIC": bic, "median_cpg": med_cpg, "percent_single":percent_single})
        return pd.concat([a, b, c], axis=1)

    def run(self):
        self.read()
        self.em_all()
        self.read_thetas()
        for model in self.config["models"]:
            res = self.calc_info(model)
            #save to file
            res.to_csv(os.path.join(self.outdir, str(self.name) + "_%s_info.csv"%model), index=False)
        stats = self.region_stats()
        stats.to_csv(os.path.join(self.outdir, str(self.name) + "_regions_stats.csv"), index=False)

class ConfusionRunner(InfoRunner):
    '''
    how each region is assigned
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_info(self, model):
        res = defaultdict(list)
        self.input_windows = []
        self.abs_windows = []
        for i, interval in enumerate(self.interval_order):
            if type(self.matrices[i]) is list and not self.matrices[i].any():
                continue
            window_list = [(0, self.matrices[i].shape[1])]
            if self.config["walk_on_list"]:
                window_list = list(do_walk_on_list(window_list, self.config["window_size"], self.config["step_size"]))
            methylation_matrix = self.matrices[i].tocsc()

            for j, (start, stop) in enumerate(window_list):
                section, sec_read_ind = clean_section(methylation_matrix, start, stop)
                if np.sum(section) == 0: #no data
                    continue
                self.abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], [(start, stop)])[0])
                self.input_windows.append((interval.chrom, interval.start, interval.end))
                high, low = self.theta_A[i][j], self.theta_B[i][j]
                lt = self.lambdas[i][:,j].flatten()
                src = self.sources[i][sec_read_ind]
                source_labels = np.array(self.labels)[src - 1]  # adjusted for index
                beta = np.vstack([calc_beta(section[np.array(source_labels) == t, :]) for t in self.config["cell_types"]])
                for k, cell in enumerate(self.config["cell_types"]):
                    reads = section[np.array(source_labels) == cell, :]
                    if model == "epistate-plus":
                        res[cell].append(epistate_plus_info(lt, high, low, reads))
                    else:
                        res[cell].append(model_to_fun[model](beta, reads))
        a = pd.DataFrame(self.input_windows, columns=["input_chrom", "input_start", "input_end"])
        b = pd.DataFrame(self.abs_windows, columns=["window_chrom", "window_start", "window_end"])
        c = pd.DataFrame(res, columns= self.cell_types)

        return pd.concat([a,b,c], axis=1)

class LeaveOneOutRunner(ConfusionRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def filter_regions(self, filt):
        self.interval_order = list(compress(self.interval_order, filt))
        self.matrices, self.cpgs, self.origins, self.sources = [list(compress(x, filt)) for x in
                                                                (self.matrices, self.cpgs, self.origins, self.sources)]


    def init_interval_labels(self):
        #load region labels
        region_labels = pd.read_csv(self.config["region_labels"], sep="\t", names=["chrom", "start", "end", "label"],
                                    usecols=[0,1,2,3], header=None)
        intervals = [str(GenomicInterval().set_from_positions(chrom, start, end)) for chrom, start, end in
                     region_labels[["chrom", "start", "end"]].to_records(index=False)]
        interval_to_label = dict(zip(intervals, region_labels["label"].values))
        labels = [interval_to_label[str(x)] for x in self.interval_order]
        filt = [x in self.cell_types for x in labels]
        self.region_labels = list(compress(labels, filt))
        self.filter_regions(filt)

    def init_windows(self):
        self.window_list = []
        self.abs_windows = []
        self.input_windows = []
        for i, interval in enumerate(self.interval_order):
            window_list = [(0, self.matrices[i].shape[1])]
            if self.config["walk_on_list"]:
                window_list = list(do_walk_on_list(window_list, self.config["window_size"], self.config["step_size"]))
            self.window_list.append(window_list)
            for j, (start, stop) in enumerate(window_list):
                self.abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], [(start, stop)])[0])
                self.input_windows.append((interval.chrom, interval.start, interval.end))


    def init_samples(self):
        self.cell_to_samples = defaultdict(list)
        for cell, sample in zip(self.config["labels"], self.config["person_id"]):
            self.cell_to_samples[cell].append(sample)

    def build_refs(self):
        '''
        build reference atlases for each region without each person
        :return:
        '''
        self.refs = []
        for i, interval in enumerate(self.interval_order):
            ref = {}
            if type(self.matrices[i]) is list and not self.matrices[i].any():
                continue
            methylation_matrix = self.matrices[i].tocsc()
            for j, (start, stop) in enumerate(self.window_list[i]):
                section, sec_read_ind = clean_section(methylation_matrix, start, stop)
                if np.sum(section) == 0: #no data
                    continue
                src = self.sources[i][sec_read_ind] # which input sample
                source_id = np.array(self.config["person_id"])[src - 1] # which person
                for person in self.cell_to_samples[self.region_labels[i]]:
                    filt = source_id != person #filter out person
                    filt_section, filt_sec_read_ind = section[filt,:], sec_read_ind[filt]
                    filt_src = self.sources[i][filt_sec_read_ind]
                    source_labels = np.array(self.labels)[filt_src - 1] # which cell types
                    beta = np.vstack(
                        [calc_beta(filt_section[np.array(source_labels) == t, :], filt_section.shape[1]) for t in self.config["cell_types"]])
                    em_results = run_em(sp.sparse.csr_matrix(filt_section), [(start, stop)])


                    source_ids = [self.label_to_id[x] for x in source_labels]
                    stats = get_all_stats(em_results["Indices"], em_results["Probs"],
                                          dict(zip(np.arange(len(source_ids)), source_ids)),
                                          len(self.cell_types), self.config["get_pp"])
                    if person not in ref:
                        ref[person] = defaultdict(list)
                    ref[person]["Betas"].append(beta)
                    ref[person]["Lambdas"].append(stats[1:, :, 4])
                    ref[person]["ThetaA"].append(em_results["Theta_A"][0])
                    ref[person]["ThetaB"].append(em_results["Theta_B"][0])
            self.refs.append(ref)

    def calc_info(self, model):
        res = []
        for i, interval in enumerate(self.interval_order):
            if type(self.matrices[i]) is list and not self.matrices[i].any():
                continue
            methylation_matrix = self.matrices[i].tocsc()
            target = self.region_labels[i]
            for j, (start, stop) in enumerate(self.window_list[i]):
                section, sec_read_ind = clean_section(methylation_matrix, start, stop)
                if np.sum(section) == 0: #no data
                    continue
                weights = []
                estimates = []
                for person in self.cell_to_samples[target]:
                    ref = self.refs[i][person]
                    beta, lt, thetaA, thetaB = ref["Betas"][j],ref["Lambdas"][j],ref["ThetaA"][j],ref["ThetaB"][j]
                    src = self.sources[i][sec_read_ind]
                    source_id = np.array(self.config["person_id"])[src - 1] #which person
                    source_labels = np.array(self.labels)[src - 1] #which cell types
                    filt = (source_id == person) & (source_labels == target) #target person, target cell type
                    reads = section[filt, :]
                    weights.append(reads.shape[0])
                    if model == "epistate-plus":
                        estimates.append(epistate_plus_info(lt.flatten(), thetaA, thetaB, reads))
                    else:
                        estimates.append(model_to_fun[model](beta, reads))
                res.append(np.average(np.vstack(estimates), weights=weights, axis=0))
        a = pd.DataFrame(self.input_windows, columns=["input_chrom", "input_start", "input_end"])
        b = pd.DataFrame(self.abs_windows, columns=["window_chrom", "window_start", "window_end"])
        c = pd.DataFrame(res, columns= self.cell_types)

        return pd.concat([a,b,c], axis=1)

    def em_all(self):
        for i, interval in enumerate(self.interval_order):
            window_list = [(0, self.matrices[i].shape[1])]
            if self.config["walk_on_list"]:
                window_list = list(do_walk_on_list(window_list, self.config["window_size"], self.config["step_size"]))
            em_results = run_em(self.matrices[i], window_list)
            stats = get_all_stats(em_results["Indices"], em_results["Probs"], dict(zip(np.arange(len(self.sources[i])), self.sources[i])),
                                  len(self.config["epiread_files"]), self.config["get_pp"])
            self.results.append(em_results)
            self.stats.append(stats)

    def region_stats(self):
        input_windows = []
        abs_windows = []
        bic = []
        med_cpg = []
        percent_single = []
        label = []
        for i, interval in enumerate(self.interval_order):
            if "windows" in self.results[i] and self.results[i]["windows"]:  # any windows with results
                abs_windows.append(relative_intervals_to_abs(interval.chrom, self.cpgs[i], self.results[i]["windows"]))
                input_windows.append([(interval.chrom, interval.start, interval.end)]*len(self.results[i]["windows"]))
                bic.extend(self.results[i]["BIC"])
                med_cpg.extend(self.results[i]["median_cpg"])
                percent_single.extend(self.results[i]["percent_single"])
                label.append(self.region_labels[i]) #TODO: fix for walk on list
        a = pd.DataFrame(self.input_windows, columns=["input_chrom", "input_start", "input_end"])
        b = pd.DataFrame(self.abs_windows, columns=["window_chrom", "window_start", "window_end"])
        c = pd.DataFrame({"BIC": bic, "median_cpg": med_cpg, "percent_single":percent_single, "label":label})
        return pd.concat([a, b, c], axis=1)

    def run(self):
        self.read()
        self.init_interval_labels()
        self.init_windows()
        self.init_samples()
        self.build_refs()
        for model in self.config["models"]:
            res = self.calc_info(model)
            #save to file
            res.to_csv(os.path.join(self.outdir, str(self.name) + "_%s_info.csv"%model), index=False)

        self.em_all() #for stats
        stats = self.region_stats()
        stats.to_csv(os.path.join(self.outdir, str(self.name) + "_regions_stats.csv"), index=False)



def calc_x_given_prob(prob, x):
    x_c_m = x == METHYLATED
    x_c_u = x == UNMETHYLATED
    log_prob = np.nan_to_num(np.log(prob).T)
    log_one_minus_prob = np.nan_to_num(np.log(1 - prob).T)
    res = (np.matmul(x_c_m, log_prob) + np.matmul(x_c_u, log_one_minus_prob)).T
    return res

def celfie_info(beta_t_m, x):
    '''

    :param x: np array with c reads, m cpgs
    :param beta_t_m: np array with t cell types, m cpgs
    :return: info per cell type t
    '''
    C, M = x.shape
    new_x = np.zeros((C*M, M))
    new_x.fill(np.nan)
    row, col = np.indices((C, M))
    new_x[np.arange(C*M).reshape(C,M), col] = x[row, col] #split to one val per row
    new_x = new_x[(np.sum(np.isnan(new_x), axis=1) < M),:] #remove empty rows
    return celfie_plus_info(beta_t_m, new_x)

def celfie_plus_info(beta_t_m, x):
    '''
    :param x: np array with c reads, m cpgs
    :param beta_t_m: np array with t cell types, m cpgs
    :return: info per cell type t
    '''
    a = np.apply_along_axis(calc_x_given_prob, axis=1, arr=beta_t_m, x=x)
    T, C = a.shape
    b = logsumexp(a, axis=0)
    log_z = a - np.tile(b, (T, 1))
    z = np.array(np.exp(log_z))
    alpha = np.sum(z, axis=1)
    alpha = alpha/np.sum(alpha)
    return alpha

def epistate_plus_info(lambda_t, theta_high, theta_low, x):
    '''

    :param lambda_t: prob of theta high given cell type t
    :param theta_high: prob of methylation per cpg
    :param theta_low: prob of methylation per cpg
    :param x:  np array with c reads, m cpgs
    :return: info per cell type t
    '''
    C, M = x.shape
    T = len(lambda_t)
    log_high = np.tile(calc_x_given_prob(theta_high, x), (T, 1)) #T,C
    log_low =  np.tile(calc_x_given_prob(theta_low, x), (T, 1)) #T,C
    lt = np.tile(lambda_t, (C, 1)).T #T, C
    log_z = logsumexp([log_high + np.log(lt), log_low + np.log(1 - lt)], axis=0) #T, C
    log_z = log_z - np.tile(logsumexp(log_z, axis=0), (T,1)) #T, C
    alpha = np.sum(np.exp(log_z), axis=1) #T
    alpha = alpha/np.sum(alpha) #T
    return alpha

def calc_beta(x, n_cols=1):
    if not x.any():
        res = np.zeros(n_cols)
        res.fill(np.nan)
        return res
    x_c_m = x == METHYLATED
    x_c_u = x == UNMETHYLATED
    beta = x_c_m.sum(axis=0)/(x_c_m.sum(axis=0)+x_c_u.sum(axis=0))
    return np.array(beta).flatten()

model_to_fun = {"celfie": celfie_info, "celfie-plus": celfie_plus_info, "epistate-plus": epistate_plus_info}

#%%
# config = {"genomic_intervals": '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/merged_tims.bed',
#   "cpg_coordinates": "/Users/ireneu/PycharmProjects/old_in-silico_deconvolution/debugging/hg19.CpG.bed.sorted.gz",
#   "epiread_files": ['/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Acinar-Z000000QX.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Acinar-Z0000043W.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Acinar-Z0000043X.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Acinar-Z0000043Y.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Alpha-Z00000453.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Alpha-Z00000456.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Alpha-Z00000459.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Beta-Z00000452.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Beta-Z00000455.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Beta-Z00000458.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Delta-Z00000451.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Delta-Z00000454.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Delta-Z00000457.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Duct-Z000000QZ.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Duct-Z0000043T.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Duct-Z0000043U.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Duct-Z0000043V.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Endothel-Z0000042D.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Endothel-Z0000042X.epiread.gz',
# '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/sorted_Pancreas-Endothel-Z00000430.epiread.gz'
#                     ],
#
#
# "region_labels":"/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/region_assignment.bed",
# "labels":["Acinar","Acinar","Acinar","Acinar","Alpha","Alpha","Alpha","Beta","Beta","Beta","Delta","Delta","Delta",
#           "Duct","Duct","Duct","Duct","Endothel","Endothel","Endothel"],
# "person_id":["AFCD035","H2226","H2224","H2220","H2207","H2211","SCICRC1146","H2207","H2211","SCICRC1146","H2207","H2211","SCICRC1146",
#              "AFCD035","H2226","H2224","H2220","UA213A","UA212","UA205"],
# "cell_types" : ["Delta", "Duct", "Acinar" , "Endothel", "Alpha", "Beta"],
#           "models": ["celfie-plus", "celfie","epistate-plus"],
#   "outdir": "/Users/ireneu/PycharmProjects/bimodal_detector/results/",
#   "epiformat": "old_epiread_A",
#   "header": False,
#   "bedfile": True,
#   "parse_snps": False,
#     "get_pp":False,
#   "walk_on_list": False,
#     "verbose" : False,
#   "window_size": 5,
#   "step_size": 1,
#           "bic_threshold":np.inf,
#     "name": "TIMs100_leave_one_out_confusion",
#   "logfile": "log.log"}
# runner = LeaveOneOutRunner(config)
# runner.run()
#%%
