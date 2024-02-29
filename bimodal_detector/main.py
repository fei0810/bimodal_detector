import click
import json
from bimodal_detector.runners import Runner, ParamEstimator, TwoStepRunner, AtlasEstimator
from bimodal_detector.region_information import InfoRunner, ConfusionRunner, LeaveOneOutRunner

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
    elif config["run_type"]=="atlas_estimation":
        runner=AtlasEstimator
    elif config["run_type"] == "info":
        runner = InfoRunner
    elif config["run_type"] == "confusion":
        runner = ConfusionRunner
    elif config["run_type"] == "leave_one_out":
        runner = LeaveOneOutRunner

    em_runner = runner(config)
    em_runner.run()

if __name__ == '__main__':
    main()
# with open("/Users/ireneu/berman_lab/Results/rrbs/rrbs_config.json") as infile:
#     config = json.load(infile)
# cell_types = ['Adipocytes', 'Colon-Fibro', 'Heart-Fibro', 'Dermal-Fibro', 'Skeletal-Musc', 'Heart-Cardio', 'Bone-Osteob', 'Oligodend',
#               'Pancreas-Duct', 'Pancreas-Acinar', 'Pancreas-Delta', 'Pancreas-Beta', 'Pancreas-Alpha', 'Thyroid-Ep',
#               'Fallopian-Ep', 'Ovary+Endom-Ep', 'Eryth-prog', 'Blood-NK', 'Blood-Granul', 'Blood-B', 'Epid-Kerat',
#            'Lung-Ep-Bron', 'Prostate-Ep', 'Bladder-Ep', 'Breast-Luminal-Ep', 'Breast-Basal-Ep', 'Lung-Ep-Alveo', 'Gallbladder']
#
# atlas_dir = "/Users/ireneu/berman_lab/Results/rrbs/atlas_pat/"
# epiread_files = [atlas_dir +f+"_atlas_epipaths.epiread.gz" for f in cell_types]
# basic_config = {"genomic_intervals": ["chr1:11594:11600","chr1:12884-12892"],
#                 "cpg_coordinates": "/Users/ireneu/PycharmProjects/deconvolution_models/tests/data/hg19_pat_cpg.bed.gz",
#                 "cell_types": cell_types,
#                 "epiread_files": epiread_files,
#                 "labels": cell_types,
#                 "outdir": "results", "name": config["name"],
#                 "epiformat": config["epiformat"], "header": False, "bedfile": False, "parse_snps": False,
#                 "get_pp": False, "walk_on_list": False, "verbose": False}
# runner = AtlasEstimator(basic_config)
# runner.run()
#

