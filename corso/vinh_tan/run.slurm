#!/bin/bash
#SBATCH --job-name=vinh_tan
#SBATCH --output=mhh-%j.out
#SBATCH --error=mhh-%j.err
#SBATCH --partition=genoa
#SBATCH -n 384
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH -t 24:00:00

source ~/setup_env.sh

srun ./microhh init vinh_tan

find . -maxdepth 1 -type f -name '*_ext*' | while read -r file; do
    newname="${file/_ext/}"
    echo "Renaming: $file -> $newname"
    mv "$file" "$newname"
done

srun ./microhh run vinh_tan
