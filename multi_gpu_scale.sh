#!/bin/bash -l
#
#SBATCH --output=./SLURM_OUT_FILES/%j_%x.out
#SBATCH -J multi_gpu_scaling
#SBATCH --time=0:59:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --export=NONE

module purge 
module load nvhpc
module load openmpi
module load likwid

# ./compile.sh

gpgpucount=$(nvidia-smi --query-gpu=gpu_name --format=csv | grep -i "nvidia" | wc -l)

for gpcount in $(seq 1 $gpgpucount)
do
    i=$((8))
    x=$((i * 1024))
    y=$((i * 1024))
    likwid-mpirun -mpi openmpi -np $gpcount -nperdomain M:1 ./build/cg_solver -s 100 -x $x -y $y --nIt 5000
done

