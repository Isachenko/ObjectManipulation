#!/bin/bash
#SBATCH --time=14:00:00 #This is one hour
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adrianalvarez15@gmail.com
#SBATCH --output=job-%j.log
#SBATCH --mem=60GB#(This is max available RAM)
cd ObjectManipulation
module load git/2.13.2-foss-2016a
module load Python/3.5.2-foss-2016a
module load tensorflow/1.2.0-foss-2016a-Python-3.5.2
module load matplotlib/1.5.3-foss-2016a-Python-3.5.2
module load libGLU/9.0.0-foss-2016a-Mesa-11.2.1
module load V-REP/3.4.0
module load FFmpeg/3.0.2-foss-2016a
python3 main.py $*
