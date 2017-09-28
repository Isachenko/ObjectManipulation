#!/bin/bash
#SBATCH --time=1:00:00 #This is one hour
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adrianalvarez15@gmail.com
#SBATCH --output=job-%j.log
#SBATCH --mem=60GB#(This is max available RAM)
module load git/2.13.2-foss-2016a
module load Python/3.5.2-foss-2016a
module load tensorflow/1.2.0-foss-2016a-Python-3.5.2
matplotlib/1.5.3-foss-2016a-Python-3.5.2
module load V-REP/3.4.0
git clone https://github.com/Isachenko/ObjectManipulation.git
python A3C_experiment.py