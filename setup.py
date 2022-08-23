import setuptools
import os
TOKEN_VALUE = os.getenv('EXPORTED_VAR_WITH_TOKEN')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bimodal_detector',
    version='0.0.1',
    author='Irene Unterman',
    author_email='irene.guberman@mail.huji.ac.il',
    description='fit two epistates to methylation sequencing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/methylgrammarlab/bimodal_detector',
    project_urls = {
        "Bug Tracker": "https://github.com/methylgrammarlab/bimodal_detector/issues"
    },
    license='MIT',
    packages=['bimodal_detector'],
    install_requires=['numpy', 'pandas', 'scipy', 'sklearn', "Click", "pytest", "pysam"],
                      # f"epiread-tools @ git+https://{TOKEN_VALUE}@github.com/methylgrammarlab/epiread-tools.git"
                      # ],
    include_package_data=True,
    entry_points={
    "console_scripts":[
    "runEM = bimodal_detector.main:main",
    ]
    },
)
# python3 setup.py sdist bdist_wheel
