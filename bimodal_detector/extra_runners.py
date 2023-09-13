from bimodal_detector.runners import ParamEstimator
from epiread_tools.naming_conventions import *
import numpy as np
import os
import gzip

def format_array(array):
    if isinstance(array, (int, float, np.float128)):
        # Handle single values by formatting them with three decimal places
        return '{:.3f}'.format(array)
    elif isinstance(array, np.ndarray):
        if array.ndim == 1:
            # Handle 1D arrays (single row) by formatting it as a single comma-delimited row
            return ','.join(map('{:.3f}'.format, array))
        elif array.ndim == 2:
            # Handle 2D arrays (list of rows) by formatting each row as a comma-delimited list
            formatted_rows = [','.join(map('{:.3f}'.format, row)) for row in array]
            return '\t'.join(formatted_rows)
    raise ValueError("Input array should be a single value, a 1D iterable, or a 2D iterable (e.g., list of lists or NumPy array).")


class OneStepRunner(ParamEstimator):
    '''
    run EM of a list of windows
    output is one row per window
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def write_output(self):
        '''
        chr start end BIC stateA stateB n_read n>0.9, n<0.1, n>0.5, pp_mean, pp_stdev
        everything from n_read onwards is per group, comma delilmited
        :return:
        '''
        chrom = []
        interval_start, interval_end = [], [] #input region
        win_start, win_end = [], [] #if walk on region
        bics = []
        stateA, stateB = [], []
        stats = []
        for i, interval in enumerate(self.interval_order):
            if "windows" in self.results[i] and self.results[i]["windows"]: #any results
                n_windows = len(self.results[i]["windows"])
                chrom.extend([interval.chrom]*n_windows)
                interval_start.extend([interval.start]*n_windows)
                interval_end.extend([interval.end]*n_windows)
                bics.extend(self.results[i]["BIC"])
                stateA.extend([format_array(x) for x in self.results[i]["Theta_A"]])
                stateB.extend([format_array(x) for x in self.results[i]["Theta_B"]])

                for j, (x, y) in enumerate(self.results[i]["windows"]):
                    win_start.append(self.cpgs[i][x])
                    win_end.append(self.cpgs[i][y-1]+1)#end of last cpg
                    win_stats = self.stats[i][:,j,:]
                    stats.append('\t'.join([format_array(row) for row in win_stats.transpose()]))


        combined_chrom = np.hstack(chrom)
        combined_interval_start = np.hstack(interval_start)
        combined_interval_end = np.hstack(interval_end)
        combined_win_start = np.hstack(win_start)
        combined_win_end = np.hstack(win_end)
        combined_bics = np.hstack(bics)

        # Combine stateA and stateB separately since they may have different lengths
        combined_stateA = np.hstack(stateA)
        combined_stateB = np.hstack(stateB)

        # Combine stats separately since it's a 1D array
        combined_stats = np.hstack(stats)

        # Column stack the combined variables
        output_array = np.column_stack((combined_chrom, combined_interval_start, combined_interval_end,
                                        combined_win_start, combined_win_end, combined_bics, combined_stateA,
                                        combined_stateB, combined_stats))

        # Define the format string for each column
        format_str = ['%s', '%d', '%d', '%d', '%d', '%.3f', '%s', '%s', '%s'] + ['%s'] * (output_array.shape[1] - 9)
        with gzip.open(os.path.join(self.outdir, str(self.name) + "_EM_results.csv.gz"), "a+") as outfile:
            np.savetxt(outfile, output_array, delimiter=TAB, fmt=format_str)




#
# config = {"cpg_coordinates": "/Users/ireneu/PycharmProjects/deconvolution_models/demo/hg19.CpG.bed.sorted.gz",
#           "bedfile":False,
#           "genomic_intervals":["chr1:1045636:1045789", "chr1:1095821-1096180"],
#           # "genomic_intervals":"/Users/ireneu/PycharmProjects/deconvolution_models/tests/data/sensitivity_200723_U250_merged_regions_file.bed",
#           "outdir":"/Users/ireneu/PycharmProjects/bimodal_detector/results",
#           "epiformat":"old_epiread_A", "header":False, "epiread_files":["/Users/ireneu/PycharmProjects/deconvolution_models/tests/data/sensitivity_200723_U250_4_rep15_mixture.epiread.gz",
#                                                                         "/Users/ireneu/PycharmProjects/deconvolution_models/tests/data/sensitivity_200723_U250_3_rep15_mixture.epiread.gz"],
#           "groups": ["banana", "apple"],
#           "atlas_file": "/Users/ireneu/PycharmProjects/deconvolution_models/tests/data/sensitivity_200723_U250_atlas_over_regions.txt",
#             "percent_u": "/Users/ireneu/PycharmProjects/deconvolution_models/tests/data/sensitivity_200723_U250_percent_U.bedgraph",
#   "num_iterations": 10, "stop_criterion": 1e-05, "random_restarts": 1, "summing":False,
#           "min_length":1, "u_threshold":0.25, "npy":False, "weights":False, "minimal_cpg_per_read":1,
#           "name":"banana", "verbose":False, "walk_on_list":True, "window_size":5, "step_size":1
#           }
#
# runner = OneStepRunner(config)
# runner.run()