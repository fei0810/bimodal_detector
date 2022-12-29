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
from collections import defaultdict
from bimodal_detector.run_em import do_walk_on_list, clean_section
import os

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
        model_to_fun = {"celfie": celfie_info, "celfie-plus":celfie_plus_info, "epistate-plus":epistate_plus_info}
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

def calc_beta(x):
    if not x.any():
        return np.nan
    x_c_m = x == METHYLATED
    x_c_u = x == UNMETHYLATED
    beta = x_c_m.sum(axis=0)/(x_c_m.sum(axis=0)+x_c_u.sum(axis=0))
    return np.array(beta).flatten()

#%%
# config = {"genomic_intervals": '/Users/ireneu/PycharmProjects/bimodal_detector/tests/data/netanel_pancreas_only.bed',
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
# "labels":["Acinar","Acinar","Acinar","Acinar","Alpha","Alpha","Alpha","Beta","Beta","Beta","Delta","Delta","Delta",
#           "Duct","Duct","Duct","Duct","Endothel","Endothel","Endothel"],
# "cell_types" : ["Delta", "Duct", "Acinar" , "Endothel", "Alpha", "Beta"],
#           "models": ["epistate-plus"],
#   "outdir": "/Users/ireneu/PycharmProjects/bimodal_detector/results/",
#   "epiformat": "old_epiread_A",
#   "header": False,
#   "bedfile": True,
#   "parse_snps": False,
#     "get_pp":False,
#   "walk_on_list": True,
#     "verbose" : False,
#   "window_size": 5,
#   "step_size": 1,
#           "bic_threshold":np.inf,
#     "name": "testing",
#   "logfile": "log.log"}
# runner = InfoRunner(config)
# runner.run()