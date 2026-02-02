#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH -J single_gpu_scaling
#SBATCH --time=0:59:00
#SBATCH --export=NONE

module purge 
module load nvhpc
module load openmpi
module load likwid

# ./compile.sh

for i in {1..12}
do
    x=$((i * 1024))
    y=$((i * 1024))
    likwid-mpirun -np 1 -nperdomain M:1 ./build/cg_solver -s 100 -x $x -y $y
done

