'''
find how informative each region is for each cell type
per model
'''
import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools/") ###
sys.path.append("/Users/ireneu/PycharmProjects/deconvolution_models/deconvolution_models") ###

from epiread_tools.naming_conventions import *
from scipy.special import logsumexp
from runners import AtlasEstimator
from collections import defaultdict


class InfoRunner(AtlasEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_thetas(self):
        self.theta_A = []
        self.theta_B = []
        for i, interval in enumerate(self.interval_order):
            if "windows" in self.results[i] and self.results[i]["windows"]:  # any windows with results
                thetas = zip(self.results[i]["Theta_A"], self.results[i]["Theta_B"])
                for m, n in thetas:
                   self.theta_A.append(m)
                   self.theta_B.append(n)

    def filter_empty_regions(self):
        self.matrices = [x for i, x in enumerate(self.matrices) if (i != 54 and i != 154)]
        lambdas = [x for i, x in enumerate(self.lambdas) if (i != 54 and i != 154)]
        theta_high = [np.array(eval(x)) for x in theta_A]
        theta_low = [np.array(eval(x)) for x in theta_B]
        ind_to_type = dict(zip(np.arange(1, 21),
                               ["Acinar", "Acinar", "Acinar", "Acinar", "Alpha", "Alpha", "Alpha", "Beta", "Beta", "Beta",
                                "Delta", "Delta", "Delta",
                                "Duct", "Duct", "Duct", "Duct", "Endothel", "Endothel", "Endothel"]))
        sources = [x for i, x in enumerate(self.sources) if (i != 54 and i != 154)]
        sources = [[ind_to_type[x] for x in y] for y in sources]

    def calc_info(self):
        epistate, plus, celfie = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(self.matrices)):
            x = self.matrices[i].todense()
            high, low = self.theta_A[i], self.theta_B[i]
            lt = self.lambdas[i].flatten()
            src = self.sources[i]
            beta = np.vstack([calc_beta(x[np.array(src) == t, :]) for t in self.config["cell_types"]])
            for j, cell in enumerate(self.config["cell_types"]):
                reads = x[np.array(src) == cell, :]
                epistate[cell].append(epistate_plus_info(lt, high, low, reads)[j])
                plus[cell].append(celfie_plus_info(beta, reads)[j])
                celfie[cell].append(celfie_info(beta, reads)[j])

    def save(self):
        pass

    def run(self):
        self.read()
        self.em_all()
        self.read_thetas()



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
    log_high = np.tile(calc_x_given_prob(theta_high, x), (1, T)).T
    log_low =  np.tile(calc_x_given_prob(theta_low, x), (1, T)).T
    lt = np.tile(lambda_t, (C, 1)).T
    log_z = logsumexp([log_high + np.log(lt), log_low + np.log(1 - lt)], axis=0)
    log_z = log_z - np.tile(logsumexp(log_z, axis=0), (T,1))
    alpha = np.sum(np.exp(log_z), axis=1)
    alpha = alpha/np.sum(alpha)
    return alpha

def calc_beta(x):
    if not x.any():
        return np.nan
    x_c_m = x == METHYLATED
    x_c_u = x == UNMETHYLATED
    beta = x_c_m.sum(axis=0)/(x_c_m.sum(axis=0)+x_c_u.sum(axis=0))
    return np.array(beta).flatten()

