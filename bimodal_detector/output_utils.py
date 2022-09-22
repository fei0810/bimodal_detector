import os
import sys
import numpy as np
sys.path.append("/Users/ireneu/PycharmProjects/epiread-tools/") ###
from epiread_tools.naming_conventions import *
import gzip
import json

def format_array(array):
    '''
    :param array: np array
    :return: array formatted for output
    '''
    return np.array2string(array, max_line_width=np.inf,separator=",", precision=3)

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

