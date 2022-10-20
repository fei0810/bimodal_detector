#############################################################
# FILE: Likelihood_and_BIC.py
# WRITER: Irene Unterman
# DESCRIPTION: Calculate likelihood for methylation reads
# based on a single epistate or a two-epistate model
# without parameter estimation
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
# #############################################################


import sys
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools")
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from epiread_tools.naming_conventions import *

SAMPLE_DATA = "toy_data.csv"


def single_epistate_ll(R_m, R_u, theta):
    '''
    :param R_m: number of methylated reads per CpG (n sized array)
    :param R_u: number of unmethylated reads per CpG(n sized array)
    :param theta: probability of methylation per CpG (n sized array)
    :return: log likelihood of reads
    '''
    likelihood_array = np.multiply(R_m, np.log(theta)) + np.multiply(R_u, np.log(1-theta))
    return np.sum(likelihood_array)

def two_epistate_ll(state_A_ll, state_B_ll):
    '''
    :param state_A_ll: log-likelihood of reads according to state 1 (m sized array)
    :param state_B_ll: log-likelihood of reads according to state 2 (m sized array)
    :return: log likelihood of reads in two state model
    '''
    return np.sum(logsumexp([state_B_ll, state_A_ll], axis = 0))

def epistate_j_ll(methylated_indicator, unmethylated_indicator,
                  mu_j, theta, max_m, WINDOW_SIZE):
    '''
    :param methylated_indicator: bool for methylation in data
    :param unmethylated_indicator: bool for unmethylation in data
    :param mu_j: probability of read belonging to state (float)
    :param theta: probability of methylation per CpG (n sized array)
    :param max_m: read depth
    :param WINDOW_SIZE: read length (n)
    :return: log likelihood of reads in the given state
    '''
    log_theta = np.tile(np.log(theta), max_m).reshape(max_m, WINDOW_SIZE)
    log_one_minus_theta = np.tile(np.log(1-theta), max_m).reshape(max_m, WINDOW_SIZE)
    data_likelihood = np.log(mu_j) + np.sum(np.multiply(methylated_indicator, log_theta) + \
                                            np.multiply(unmethylated_indicator, log_one_minus_theta), axis = 1)
    return np.array(data_likelihood)

def is_ASM(single_ll, two_state_ll, max_m, WINDOW_SIZE): ###should max_m only include reads with data?
    '''
    :param single_ll: log likelihood of reads according to single state model
    :param two_state_ll: log likelihood of reads according to two-state model
    :param max_m: read depth
    :param WINDOW_SIZE: read length (n)
    :return: Is there allele specific methylation in the given window
    '''
    BIC_single = WINDOW_SIZE*np.log(max_m)-2*single_ll
    BIC_double = 2*WINDOW_SIZE*np.log(max_m)-2*two_state_ll
    return BIC_double < BIC_single #lower means better

def get_BIC(WINDOW_SIZE, max_m, single_ll, two_state_ll):###should max_m only include reads with data?
    '''
    :param WINDOW_SIZE: N CpGs in window
    :param max_m: maximal coverage (N reads)
    :param single_ll: likelihood according to single epistate
    :param two_state_ll: likelihood according to two epistates
    :return: Bayesian Information Criteriion
    '''
    BIC_single = WINDOW_SIZE * np.log(max_m) - 2 * single_ll
    BIC_double = 2 * WINDOW_SIZE * np.log(max_m) - 2 * two_state_ll
    return BIC_double - BIC_single


def max_single_epistate_ll(data):
    '''
    :param data: mxn array of methylation reads
    :return: estimates optimal theta for single epistate
    '''
    R_m = np.count_nonzero(data == METHYLATED, axis=0)+pseudocount
    R_u = np.count_nonzero(data == UNMETHYLATED, axis=0)+pseudocount
    theta_m = R_m/(R_m+R_u)
    return single_epistate_ll(R_m, R_u, theta_m)

def main():
    '''
    run sample dara, get likelihoods for single/two epistates
    use BIC to determine if the region is differentialy methylated
    '''
    data = pd.read_csv(SAMPLE_DATA, header=None)
    max_m, WINDOW_SIZE = data.shape
    initial_low = 0.2
    initial_high = 0.8
    theta_0 = np.array([initial_low] * WINDOW_SIZE)
    theta_1 = np.array([initial_high] * WINDOW_SIZE)
    mu_1 = 0.5
    m_ri_k = data == METHYLATED #methylation indicator
    u_ri_k = data == UNMETHYLATED #unmethylated indicator
    single_ll = max_single_epistate_ll(data)
    epistate_1_ll = epistate_j_ll(m_ri_k, u_ri_k, mu_1, theta_1, max_m, WINDOW_SIZE)
    epistate_2_ll = epistate_j_ll(m_ri_k, u_ri_k, 1-mu_1, theta_0, max_m,WINDOW_SIZE)
    two_state_ll = two_epistate_ll(epistate_1_ll, epistate_2_ll)
    print(np.exp(single_ll), np.exp(two_state_ll), is_ASM(single_ll, two_state_ll,  max_m, WINDOW_SIZE))

if __name__ == "__main__":
    main()