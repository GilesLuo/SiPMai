import os
from setuptools import find_packages, setup


def get_version() -> str:
    init = open(os.path.join("SiPMai", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires():
    return [
        "numpy>=1.16,<1.24",
        "torch>=1.4.0",
        "packaging",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "pandas",
        "opencv-python",
        "numba",
        "rdkit",
        "ray",
    ]


def get_extras_require():
    req = {
        "dev": [
            "flake8",
            "pytest",
            "pytest-cov",
            "mypy",
        ],
    }
    return req


setup(
    name='SiPMai',
    version=get_version(),
    description='A Simple Yet Effective Scanning Tunnel Microscope Image Simulator',
    author='Zhiyao Luo, Yaotian Yang, Jiali Li',
    author_email='zhiyao.luo@eng.ox.ac.uk',
    url='http://github.com/GilesLuo/SiPMai',
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    entry_points={
        'console_scripts': ['generate_pubchem=SiPMai.gen_data.gen_all_data_pipeline:main'],
        # 'console_scripts': ['test2=SiPMai.test2:main'],
    },
    install_requires=get_install_requires(),
    package_data={'SiPMai': ['smiles/*.json']},
    include_package_data=True,
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires='>=3.6, <3.10',
    keywords=["Chemistry Simulation", "STM Image Synthesis"],
    extras_require=get_extras_require(),
    project_urls={
        "Source Code": "https://github.com/my_username/my_package", # todo
    }
)
