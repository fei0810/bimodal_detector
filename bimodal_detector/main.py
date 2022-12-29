import click
import json
from runners import Runner, ParamEstimator, TwoStepRunner, AtlasEstimator
from region_information import InfoRunner

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

    em_runner = runner(config)
    em_runner.run()

if __name__ == '__main__':
    main()
