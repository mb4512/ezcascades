[![pull requests](https://img.shields.io/github/issues-pr/mb4512/ezcascades.svg)](https://github.com/mb4512/ezcascades/pulls)
[![issues](https://img.shields.io/github/issues/mb4512/ezcascades.svg)](https://github.com/mb4512/ezcascades/issues/)
[![l## icense: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)

# ezcascades - atomic descriptors

**! This is an experimental branch !**

**! Scroll below to section "Changes to ezcascades-gpu.py" for a summary of chanes!**

# ezcascades

LAMMPS Python script for simulating high-dose irradiation damage

Full details of the method are available at:
* Simultaneous cascades for large-scale high dose simulation: [10.1103/PhysRevMaterials.6.063601](https://doi.org/10.1103/PhysRevMaterials.6.063601)
* Varying recoil energies in different materials:             [10.1038/s41598-022-27087-w](https://doi.org/10.1038/s41598-022-27087-w)
* Cascades with recoil energies drawn from a spectrum:        [10.1038/s43246-024-00655-5](https://doi.org/10.1038/s43246-024-00655-5)

See the [project Wiki](../../wiki) for detailed information on features, installation, file structures, and working examples.

## Features

The ezcascades repository contains collection of Python scripts for driving the [LAMMPS Molecular Dynamics Simulator](https://github.com/lammps/lammps), compiled as a shared library, in order to simulate the evolution of microstructure under the influence of irradiation, simulated in the form of highly energetic atomic recoils. Simulation settings such as file paths, interatomic potential, system size, etc., are specified in json files, allowing for a quick and repeatable simulation setup. 

These scripts require LAMMPS to be built with the EXTRA-FIX and MANYBODY packages, for electronic stopping and embedded atom model potentials. LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to set this up.

The scripts support simulating single element materials and alloys, starting from the pristine crystal or a supplied structure, at thermal or athermal conditions, under displacement or stress constraints. The collection includes scripts for simulating large-scale high-dose microstructures through initialisation of multiple non-overlapping recoils per cascade iteration, and also scripts for simulating single-cascade events for the purpose of gathering defect generation statistics.

The collection contains the following scripts for high-dose simulations:
* [ezcascades.py: High-dose simulations via overlap collision cascades](../../wiki/High‐dose:-collision-cacades)
* [ezcra.py: High-dose simulations via creation relaxation algorithm](../../wiki/High‐dose:-Creation-relaxation-algorithm)

The collection also contains single-cascade scripts:
* [geted.py: Single-cascade simulations for obtaining displacement threshold statistics](../../wiki/Single-cascade:-Displacement-threshold-statistics)
* [getcasc.py: Single-cascade simulations for obtaining Frenkel pair generation statistics](../../wiki/Single-cascade:-Defect-generation-statistics)
* [getmelt.py: Single-cascade simulations for obtaining cascade melt size statistics](../../wiki/Single-cascade:-Melt-size-statistics)

## Getting started

To get started, you need to build LAMMPS with the EXTRA-FIX package for electronic stopping, and optionally with the MANYBODY package to enable embedded atom model potentials (or other for machine-learned potentials). LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to set this up. To avoid compatability issues, the Python environment should be set up with the same modules (compilers, MPI version, ...) used for compiling the LAMMPS shared library. In many HPC systems, you can set up the Python environment locally as a virtual environment, installing the neccessary package via the `pip` command. 

Once your environment is set up, clone this repository
```shell
$ git clone https://github.com/mb4512/ezcascades.git
```

You can test if your Python, mpi4py, LAMMPS environment is working with the supplied script:
```shell
$ mpirun -n 4 python test.py
```

Your output should show something along the lines:
```
From Python: I am rank 0 out of 4.
From Python: I am rank 2 out of 4.
From Python: I am rank 3 out of 4.
From Python: I am rank 1 out of 4.
LAMMPS (29 Aug 2024 - Development - patch_29Aug2024-704-g6d0e633edf)
WARNING: Using I/O redirection is unreliable with parallel runs. Better to use the -in switch to read input files. (../lammps.cpp:571)
OMP_NUM_THREADS environment is not set. Defaulting to 1 thread. (../comm.cpp:99)
  using 1 OpenMP thread(s) per MPI task
From LAMMPS: Hello World. You should see this message only once.
Total wall time: 0:00:00
```
Python should report the correct number of ranks, and LAMMPS should only print out once irrespective of the number of ranks. Some HPC systems do not permit executing parallel processes on the login node. In that case, you can test the configuration by submitting this script as a batch job.

The scripts in this repository assume a standard LAMMPS compilation. If you compiled your LAMMPS shared library under a different name, or you want to use accelerators (INTEL, GPU, ...), you will have to update the LAMMPS instantiations as appropriate. Some example instantiations are below:
```
lmp = lammps("gpu", cmdargs=["-sf", "gpu", "-pk", "gpu 1"])    # GPU package
lmp = lammps(cmdargs=["-pk", "intel 0 omp 2", "-sf", "intel"]) # INTEL package
```

## Changes to ezcascades-gpu.py

The script `ezcascades-gpu.py` exports voxelised atomic descriptors after each cascade description. The script now also takes in an additional command line flag to set the number of GPUs per task:
```
mpirun -n 4 ezcascades-gpu.py json/fe_gpu_example.json -ngpu 1
```
See also the script `./jobs/fe_gpu_example.job` for an example job submission script for the PITAGORA cluster, using 4 MPI threads per GPU. Besides performing overlap cascade simulations, this script also exports voxelised atomic descriptors to the `./data/job_name` directory, for example:
```
fe_gpu_example.1.dsc
fe_gpu_example.2.dsc
fe_gpu_example.3.dsc
...
```
One descriptor file is exported after each cascade iteration. The format is as stated in the file. Note the new settings in `json/fe_gpu_example.json` that control the descriptor evaluation, in particular: `average_over_fs`, which is the number of femtoseconds over which atomic trajectories are averaged before descriptors are computed, `voxelgrid`, which is the number of voxels in x, y, and z direction (aim for 10,000 atoms per voxel), and `descriptor_computes`, which lists the LAMMPS computes by which the descriptors are defined (the first entry, 'D', 'dD', 'ddD', is an identifying label set by the user and can be any string, while the following string specifies the LAMMPS command defining the corresponding compute).

## Setting up LAMMPS on PITAGORA

The below commands are for setting up LAMMPS as a shared library for a virtual environment in Python on the PITAGORA cluster. This installs the SNAP package for evaluating atomic descriptors and the GPU package. 

```
module load cmake
module load py-mpi4py/3.1.5--openmpi--4.1.6--oneapi--2024.1.0
module load cuda/12.6

cd ~
python -m venv pycuda
source pycuda/bin/activate

source pycuda/bin/activate
pip install --upgrade pip
pip install numpy scipy

cd ~
mkdir git
git clone https://github.com/lammps/lammps.git lammps
cd lammps
mkdir build

module load cmake
cmake -D PKG_ML-SNAP=ON -D PKG_EXTRA-FIX=ON -D PKG_MANYBODY=ON -D PKG_REPLICA=ON -D PKG_GPU=ON -D GPU_API=CUDA -D BUILD_OMP=ON -D LAMMPS_MACHINE=gpu-cuda ../cmake
cmake --build . -j 8

cmake -D PKG_ML-SNAP=ON -D PKG_EXTRA-FIX=ON -D PKG_MANYBODY=ON -D PKG_REPLICA=ON -D PKG_GPU=ON -D GPU_API=CUDA -D BUILD_OMP=ON -D LAMMPS_MACHINE=gpu-cuda -D BUILD_SHARED_LIBS=yes ../cmake
cmake --build . -j 8
cmake --build . --target install-python
```

## Further information

See the [project Wiki](../../wiki) for detailed information on features, file structures, and working examples.


