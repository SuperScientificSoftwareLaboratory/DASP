# DASP
Specific Dense Matrix Multiply-Accumulate Units Accelerated General Sparse Matrix-Vector Multiplication


## Introduction

Sparse matrix-vector multiplication (SpMV) plays a key role in computational science and engineering, graph processing and machine learning applications. In this work, we propose DASP, a new algorithm using specific dense MMA units for accelerating the compute part of general SpMV. We analyze the row-wise distribution of nonzeros and group the rows into three categories containing long, medium and short rows, respectively. We then organize them into small blocks of proper sizes to meet the requirement of MMA computation. For the three categories, DASP offers different strategies to complete SpMV by efficiently utilizing the MMA units.

## Installation

To better reproduce experiment results, we suggest an NVIDIA GPU with compute capability 8.0. DASP evaluation requires the CUDA GPU driver, the nvcc CUDA compiler, and the cuSPARSE library, all of them are included with the CUDA Toolkit. 

## Execution of DASP

Our test programs currently support input files encoded using the matrix market format. All matrix market datasets used in this evaluation are publicly available from the SuiteSparse Matrix Collection.

1. The command 'make xxx' generates an executable file.

`make double`

`make half`

2. Run code on matrix data. Running the program requires one parameter: matrix path.

`./spmv_double matrix.mtx`

3. Example

`cd test`

`sh run_double.sh`

## Contact us

If you have any questions about running the code, please contact Yuechen Lu.

E-mail: yuechenlu@student.cup.edu.cn
