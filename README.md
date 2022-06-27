# Visual Determination of Lane Type for Micromobility

This repo contains the scripts needed to train, validate, deploy and test the models developed in Visual Determination of Lane Type for Micromobility.

## Installation

To run the Pyhton scripts you should use Python 3.9 or above. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all dependencies of the project.

```bash
pip install -r requirements.txt
```

## Usage

The best way to run the scripts in CALCULA is to send a bash script using `sbatch`. In the folder batch_runs we have some .sh scripts useful to do so. this scripts take one argument which is the configuration ID used for that run.

All scripts in batch runs are something like this:

```bash
#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -o "outputs/train-"$1"-%j.out"
#SBATCH --mem=16G
#SBATCH -c2
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:1,gpumem:16G
echo "running $1"
python src/main.py -conf configs/$1.json -train
```

This script actually sends another script substituting `$1` by the argument passed. To send the script to the queue you should run something like this:

```bash
sbatch batch_runs/test.sh efficient_x3d_xs_1s6f_cnt
```

### Available Configurations

In the configuration folder we have many .json files storing different configurations. The format of the names are as follows:

`<model name>_<size>_<clip duration>s<frame rate>f_<crop>.json`

The models available are: efficient x3d with sizes XS and S, movinet with sizes A0 and A1. The clip duration used is tipically 1s and the frame rate is 6 or 12. The crop strategy can be cnt or dwn (remember cnt is better). New configurations can be created following this name convention and mantaining the parameters, only changing the values.
