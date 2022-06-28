#!/bin/bash
#SBATCH --mem=4G
#SBATCH -c2
#SBATCH -t 1-00:00
python src/build_annotation.py -root /imatge/morros/work_fast/mobilitat/ridesafe/barcelona/split_videos/downsampled/ -dest annotation -partition
