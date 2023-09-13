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

import os
import sys
import numpy as np
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools/") ###
from epiread_tools.naming_conventions import *
from itertools import compress
from collections import defaultdict
import gzip
import json

def format_array(array):
    '''
    :param array: np array
    :return: array formatted for output
    '''
    return np.array2string(array, max_line_width=np.inf,separator=",", precision=3, threshold=np.inf, suppress_small=True)

def relative_intervals_to_abs(chrom, cpgs, rel_intervals):
    '''
    convert relative intervals to genomic coordinates
    :param chrom: chromosome name
    :param cpgs: list of abs coordinates
    :param rel_intervals: list of intervals
    :return: array with abs coords
    '''
    res = np.zeros((len(rel_intervals),3), dtype=object)
    res[:,0] = chrom
    rel_start = np.array([x[0] for x in rel_intervals])
    rel_end = np.array([x[1] for x in rel_intervals])
    rel_end -= 1 #to get index of last cpg included
    res[:,1] = cpgs[rel_start]
    res[:,2] = cpgs[rel_end] + 1

    return res

def cpg_positions_in_interval(cpgs, rel_intervals):
    '''
    positions of all cpgs relative to the first
    :param cpgs: list of abs coordinates
    :param rel_intervals: list of intervals
    :return:
    '''
    res = []
    for start, end in rel_intervals:
        abs = cpgs[start:end]
        positions = abs - abs[0]
        res.append(positions)
    return res

def filter_list(list_to_filter, bool_filter):
    return list(compress(list_to_filter, bool_filter))

def filter_em_results(em_results, list_filter, indices=False):
    new_dict = defaultdict()
    for k, v in em_results.items():
        if indices:
            new_dict[k] = list(map(list(v).__getitem__, list_filter))
        else:
            new_dict[k] = filter_list(v, list_filter)
    return new_dict

def is_empty(stats):
    return all(map(is_empty,stats)) if isinstance(stats, list) else False