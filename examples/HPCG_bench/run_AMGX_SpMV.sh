#!/usr/local/bin/bash

##Resources
#SBATCH --job-name=HPCG_AMGX_SpMV
#SBATCH --account=a-g34
#SBATCH --output AMGX_SpMV_Output.out
#SBATCH --time 01:00:00
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1

module purge

module load cuda
module load cmake
module load gcc
module load cray-mpich

cd /users/dknecht/amgx_expanded_for_HPCG

# remove the old build if it exists
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build

cd build

# # Build the project
cmake ..
make -j16

cd examples

srun ./hpcg_bench
