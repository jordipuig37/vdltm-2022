#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH -o "outputs/test-"$1"-%j.out"
#SBATCH --mem=10G
#SBATCH -c2
#SBATCH -t 1-00:00
echo "running $1"
python src/main.py -conf configs/$1.json -test
