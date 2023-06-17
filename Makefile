#compilers
CC=/usr/local/cuda-12.0/bin/nvcc

NVCC_FLAGS = -O3 -ccbin /usr/local/gcc-12.2/bin -m64 -gencode arch=compute_80,code=sm_80

# #ENVIRONMENT_PARAMETERS
# CUDA_INSTALL_PATH = /usr/local/cuda-12.0

CUDA_LIBS = -lcusparse -lcublas
LIBS =  -lineinfo $(CUDA_LIBS)

#options
OPTIONS = -Xcompiler -fopenmp-simd

double:
	$(CC) $(NVCC_FLAGS) src/main_f64.cu -o spmv_double  -D f64 $(OPTIONS) $(LIBS) 

half:
	$(CC) $(NVCC_FLAGS) src/main_f16.cu -o spmv_half $(OPTIONS) $(LIBS) 

clean:
	rm -rf spmv_double
	rm -rf spmv_half
	rm data/*.csv