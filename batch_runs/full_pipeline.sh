#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -o "outputs/fullpipe-"$1"-%j.out"
#SBATCH --mem=10G
#SBATCH -c2
#SBATCH -t 1-00:00
#SBATCH -w gpic12
#SBATCH --gres=gpu:1,gpumem:10G
echo "running $1"
python src/main.py -conf configs/$1.json -train -deploy -test
