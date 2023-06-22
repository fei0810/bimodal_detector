#############################################################
# FILE: expectation_maximization.py
# WRITER: Irene Unterman
# DESCRIPTION: Estimate parameters for two-epistate model
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
sys.path.append("/Users/ireneu/PycharmProjects/bimodal_detector")

import numpy as np
import pandas as pd
import math
from scipy.special import logsumexp
from bimodal_detector.Likelihood_and_BIC import *
from epiread_tools.naming_conventions import *

class two_epistate_EM:

    def __init__(self, data, theta_A, theta_B, mu_A, convergence_thr, max_iteration, read_from_file = False, verbose = False,
                 use_theta_convergence=False, theta_thresh=0.001):
        self.verbose = verbose
        if read_from_file:
            self.data = pd.read_csv(data, header=None)
        else:
            self.data = data
        self.max_m, self.window_size = self.data.shape
        self.mu_A = mu_A

        #indicators
        self.m_indicator = (self.data == METHYLATED).astype(int) # methylation indicator
        self.u_indicator = (self.data == UNMETHYLATED).astype(int) # unmethylated indicator
        # self.val_indicator = 1 - np.isnan(self.data).astype(int)
        self.val_indicator = 1 - (self.data == NOVAL).astype(int)

        #convergence
        self.iteration = 0
        self.max_iterations = max_iteration
        self.threshold = convergence_thr
        self.theta_threshold = theta_thresh
        self.use_theta_convergence = use_theta_convergence

        #theta
        self.previous_theta = self.previous_ll = None
        self.current_theta = self.initialize_thetha(theta_A, theta_B)
        self._current_ll = None


    def initialize_thetha(self, theta_A, theta_B):
        '''
        :param theta_A: initial guess for theta of state A
        :param theta_B:initial guess for theta of state B
        :return: initial guess for theta as uniform array
        '''
        theta_methylated = np.array([np.longfloat(theta_A)] * self.window_size)
        theta_unmethylated = np.array([np.longfloat(theta_B)] * self.window_size)
        return (theta_methylated, theta_unmethylated)

    def get_ll(self):
        '''
        :return: log-likelihood according to two state model
        '''
        state_A_ll = epistate_j_ll(self.m_indicator, self.u_indicator,
                                   self.mu_A, self.current_theta[0], self.max_m, self.window_size)
        state_B_ll = epistate_j_ll(self.m_indicator, self.u_indicator,
                                   1 - self.mu_A, self.current_theta[1], self.max_m, self.window_size)
        joint_ll = two_epistate_ll(state_A_ll, state_B_ll)
        if self.verbose:
            print("iteration", self.iteration, "mu A", self.mu_A, "theta", self.current_theta, "ll", joint_ll)
        return joint_ll

    def expectation(self):
        '''
        :return: estimates the probability for each read
        belonging to each state
        '''
        state_A_ll = epistate_j_ll(self.m_indicator, self.u_indicator,
                                   self.mu_A, self.current_theta[0], self.max_m, self.window_size)
        state_B_ll = epistate_j_ll(self.m_indicator, self.u_indicator,
                                   (1 - self.mu_A), self.current_theta[1], self.max_m, self.window_size)

        self.p_A_i = np.exp(np.array(state_A_ll - logsumexp([state_A_ll, state_B_ll], axis=0))) #probability of state A
        self.p_B_i = np.exp(np.array(state_B_ll - logsumexp([state_A_ll, state_B_ll], axis = 0)))# probability of state B

    def estimate_mu(self):
        '''
        :return: estimation of proportion between states
        '''
        mu_A_estimate = ((np.sum(self.p_A_i)) + pseudocount) / (self.max_m + 2 * pseudocount)
        return mu_A_estimate

    def maximization(self):
        '''
        :return: update theta according to the expectation step
        '''
        pA = np.repeat(self.p_A_i, self.window_size).reshape(self.max_m, self.window_size)
        pB = np.repeat(self.p_B_i, self.window_size).reshape(self.max_m, self.window_size)
        pA = np.multiply(pA,self.val_indicator)
        pB = np.multiply(pB,self.val_indicator)
        theta_A = np.array(np.divide(np.sum(np.multiply(self.m_indicator.astype(int),pA), axis = 0) + pseudocount,
                                  (np.sum(pA, axis = 0))+2*pseudocount))
        theta_B = np.array(np.divide(np.sum(np.multiply(self.m_indicator.astype(int),pB), axis = 0)+pseudocount,
                                     (np.sum(pB, axis = 0))+2*pseudocount))
        return theta_A, theta_B

    def test_convergence(self):
        '''
        :return: True if EM converged.
        maybe easier/faster to look at theta update instead of ll
        '''
        self.update_likelihoods()
        if self.previous_ll is None:
            return False
        assert self._current_ll is not None and not np.isnan(self._current_ll)
        assert self._current_ll >= self.previous_ll or math.isclose(self._current_ll, self.previous_ll, rel_tol=0.001)
        #Pseudocounts may slightly affect likelihood, which is why we use math.isclose()
        return self._current_ll - self.previous_ll < self.threshold

    def quick_convergence_test(self):
        '''
        :return: True if EM converged.
        '''
        if self.previous_theta is None:
            return False
        return (self.current_theta[0] - self.previous_theta[0] < self.theta_threshold).all() \
        and (self.current_theta[1] - self.previous_theta[1] < self.theta_threshold).all()

    def rescale_mu(self):
        '''
        only relevant if mu is fixed at 0.5 and estimate_mu is not called
        '''
        mixing = 0.5
        estimated_mu = self.estimate_mu()
        if estimated_mu > mixing:
            self.p_A_i *= mixing / estimated_mu
            self.p_B_i = 1 - self.p_A_i
        else:
            adjustment = mixing/(1-estimated_mu)
            self.p_A_i = 1 - self.p_A_i
            self.p_A_i*= adjustment
            self.p_A_i = 1 - self.p_A_i
            self.p_B_i = 1 - self.p_A_i

    def update_likelihoods(self):
        self.previous_ll = self._current_ll
        self._current_ll = self.get_ll()

    def run_em(self):
        '''
        :return: esimate theta to maximize likelihood
        '''
        if self.use_theta_convergence:
            convergence_test = self.quick_convergence_test
        else:
            convergence_test = self.test_convergence
        while self.iteration <= self.max_iterations and not convergence_test():
            self._current_ll = None
            self.iteration += 1

            self.expectation()

            #save old thetha
            self.previous_theta = self.current_theta

            # create new mu
            self.mu_A = self.estimate_mu()
            # self.rescale_mu() - if estimate_mu isn't used

            #create new theta
            self.current_theta = self.maximization()

            #recalculate likelihood
        return self.current_theta
    @property
    def current_ll(self):
        if self._current_ll is None:
            self._current_ll = self.get_ll()
        return self._current_ll

    def get_read_labels(self):
        '''
        :return: retrieve most likely assignment per read
        '''
        return np.argmax([self.p_B_i, self.p_A_i], axis = 0)

    def get_posterior_probability(self):
        '''
        :return: retrieve the probability for the likelier
        assignment per read
        '''
        return self.p_A_i

    def get_thetas(self):
        '''
        :return: theta estimations
        '''
        return self.current_theta

    def get_mu_estimate(self):
        '''
        :return: the current estimate of mu
        prior on state A
        '''
        return self.mu_A
#