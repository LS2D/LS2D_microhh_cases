#!/bin/bash
#SBATCH --job-name=mmij
#SBATCH --output=mhh-%j.out
#SBATCH --error=mhh-%j.err
#SBATCH --partition=par
#SBATCH -n 256
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH -t 12:00:00

source ~/setup_env.sh

# Get case name from .ini file.
ini_file=$(ls *.ini)
case_name="${ini_file%.ini}"

# Init phase for grid/fft/etc. files.
srun ./microhh init $case_name

# Overwrite restart files and base state density.
find . -maxdepth 1 -type f -name '*_ext*' | while read -r file; do
    newname="${file/_ext/}"
    echo "Renaming: $file -> $newname"
    mv "$file" "$newname"
done

# Run case.
srun ./microhh run $case_name

# Post-processing.
source ~/venvs/default/bin/activate
python cross_to_nc.py -n 32 -p single
