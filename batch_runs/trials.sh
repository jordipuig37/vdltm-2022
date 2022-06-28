#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -o "outputs/trials-"$1"-%j.out"
#SBATCH --mem=16G
#SBATCH --nodelist gpic12
#SBATCH -c2
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:1,gpumem:10G
echo "running $1"
python src/main.py -conf configs/$1.json -trials
