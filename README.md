# SiPMai: A Simple Yet Effective Scanning Probe Microscope Auto Image Generator for Deep Learning

This project provides a streamlined pipeline for generating and handling molecular data, specifically for use in machine learning models. The toolkit involves generating SMILES representations, using ray tracing to create images, and preparing dataset indices for training, validation, and testing sets.

## Table of Contents

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Credits](#credits)
5. [License](#license)

## Project Description

This repository contains several Python scripts that together form a pipeline for the generation and management of molecular data. Specifically, it includes:

1. `gen_data/smile_generation.py`: A script for generating SMILES representations of molecules. It requires a CSV file containing molecule data as input and produces a JSON file containing the generated SMILES strings.
2. `gen_data/ray_generation.py`: A script that uses ray tracing to generate images of the molecules described by the SMILES strings. It has several options for customization, such as resolution, blur, and the use of motion blur and gaussian noise.
3. `gen_data/prepare_dataset.py`: A script that creates indices for the generated molecules and splits them into training, validation, and testing sets. It creates JSON files containing these indices.

The scripts are designed to be used in sequence, but can also be used independently if needed.

## Installation

This project is written in Python and requires the following Python libraries:

Please note that Python > 3.10 is not supported (due to Ray).

```
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
```

You can install these libraries using pip:

```bash
pip install SiPMai
```

or build from source:

Open your terminal and execute the following command:

```sh
git clone https://github.com/GilesLuo/SiPMai.git
cd SiPMai
python setup.py install
```

## Usage

### Data generation:

In a ternimal, do:

```
generate_pubchem
```

It will generate a 100k dataset for molecules with 39<=`num_atom`<=200 in the **command execution directory**.

You may modify the generation configuration by doing:

```
generate_pubchem --your_args
```

Please refer to `SiPMai/gen_data/gen_all_data_pipeline.py` for a complete list of arguments.

Equivalently, you can call the main() function directly from a python script, such as:

```
import SiPMai
from SiPMai.gen_data.gen_all_data_pipeline import gen_all_data, main

main()  # generate with preset arguments

# or 

from SimpTM.gen_data.gen_all_data_pipeline import gen_all
gen_all_data(many_args)   # generate with user-defined arguments
```

### Loading Data

We also provide a Pytorch DataLoader template to load the generated datasets. Details please refer to `SiPMai/utils/dataloader`.

More features are under development. Please feel free to raise issues and participate in developing this tool.

## Credits

This project was made possible thanks to the contributions of the team members and the use of multiple open-source libraries.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
