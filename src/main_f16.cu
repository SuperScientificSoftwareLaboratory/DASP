#include "dasp_f16.h"

int verify_new(MAT_VAL_TYPE *cusp_val, MAT_VAL_TYPE *cuda_val, int *new_order, int length)
{
    for (int i = 0; i < length; i ++)
    {
        int cusp_idx = new_order[i];
        float temp_cusp_val = cusp_val[cusp_idx];
        float temp_cuda_val = cuda_val[i];
        if (fabs(temp_cusp_val - temp_cuda_val) > 1)
        {
            printf("error in (%d), cusp(%4.2f), cuda(%4.2f),please check your code!\n", i, temp_cusp_val, temp_cuda_val); 
            return -1;
        }
    }
    printf("Y(%d), compute succeed!\n", length);
    return 0;
}

__host__
void cusparse_spmv_all(MAT_VAL_TYPE *cu_ValA, MAT_PTR_TYPE *cu_RowPtrA, int *cu_ColIdxA, 
                       MAT_VAL_TYPE *cu_ValX, MAT_VAL_TYPE *cu_ValY, int rowA, int colA, MAT_PTR_TYPE nnzA,
                       long long int data_origin1, long long int data_origin2, double *pre_time, double *cu_time, double *cu_gflops, double *cu_bandwidth1, double *cu_bandwidth2)
{
    struct timeval t1, t2;

    MAT_VAL_TYPE *dA_val, *dX, *dY;
    int *dA_cid;
    MAT_PTR_TYPE *dA_rpt;
    float alpha = 1.0, beta = 0.0;

    cudaMalloc((void **)&dA_val, sizeof(MAT_VAL_TYPE) * nnzA);
    cudaMalloc((void **)&dA_cid, sizeof(int) * nnzA);
    cudaMalloc((void **)&dA_rpt, sizeof(MAT_PTR_TYPE) * (rowA + 1));
    cudaMalloc((void **)&dX, sizeof(MAT_VAL_TYPE) * colA);
    cudaMalloc((void **)&dY, sizeof(MAT_VAL_TYPE) * rowA);

    cudaMemcpy(dA_val, cu_ValA, sizeof(MAT_VAL_TYPE) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_cid, cu_ColIdxA, sizeof(int) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_rpt, cu_RowPtrA, sizeof(MAT_PTR_TYPE) * (rowA + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, cu_ValX, sizeof(MAT_VAL_TYPE) * colA, cudaMemcpyHostToDevice);
    // cudaMemset(dY, 0.0, sizeof(MAT_VAL_TYPE) * rowA);

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer = NULL;
    size_t               bufferSize = 0;

    gettimeofday(&t1, NULL);
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, rowA, colA, nnzA, dA_rpt, dA_cid, dA_val,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    cusparseCreateDnVec(&vecX, colA, dX, CUDA_R_16F);
    cusparseCreateDnVec(&vecY, rowA, dY, CUDA_R_16F);
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    gettimeofday(&t2, NULL);
    *pre_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // printf("cusparse preprocessing time: %8.4lf ms\n", *pre_time);

    for (int i = 0; i < 100; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);
    for (int i = 0; i < 1000; ++i)
    {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    *cu_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / 1000;
    *cu_gflops = (double)((long)nnzA * 2) / (*cu_time * 1e6);
    *cu_bandwidth1 = (double)data_origin1 / (*cu_time * 1e6); 
    *cu_bandwidth2 = (double)data_origin2 / (*cu_time * 1e6); 
    printf("cusparse:%8.4lf ms, %8.4lf Gflop/s, %9.4lf GB/s, %9.4lf GB/s\n", *cu_time, *cu_gflops, *cu_bandwidth1, *cu_bandwidth2);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    cudaMemcpy(cu_ValY, dY, sizeof(MAT_VAL_TYPE) * rowA, cudaMemcpyDeviceToHost);

    cudaFree(dA_val);
    cudaFree(dA_cid);
    cudaFree(dA_rpt);
    cudaFree(dX);
    cudaFree(dY);
}

__host__
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Run the code by './spmv_half matrix.mtx'.\n");
        return 0;
    }

    // struct timeval t1, t2;
    int rowA, colA;
    MAT_PTR_TYPE nnzA;
    int isSymmetricA;
    // float *csrValA_f32;

    MAT_VAL_TYPE *csrValA;
    int *csrColIdxA;
    MAT_PTR_TYPE *csrRowPtrA;

    char *filename;
    filename = argv[1];
    // int NUM = atoi(argv[2]);
    // int block_longest = atoi(argv[3]);
    int NUM = 4;
    int block_longest = 256;
    double threshold = 0.75;

    printf("\n===%s===\n\n", filename);

    mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA, &csrRowPtrA, &csrColIdxA, &csrValA, filename);
    MAT_VAL_TYPE *X_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * colA);
    initVec(X_val, colA);
    initVec(csrValA, nnzA);

    MAT_VAL_TYPE *dY_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    MAT_VAL_TYPE *Y_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    int *new_order = (int *)malloc(sizeof(int) * rowA);
    
    double pre_time = 0, cu_time = 0, cu_gflops = 0, cu_bandwidth1 = 0, cu_bandwidth2 = 0;
    long long int data_origin1 = (nnzA + colA + rowA) * sizeof(MAT_VAL_TYPE) + nnzA * sizeof(int) + (rowA + 1) * sizeof(MAT_PTR_TYPE);
    long long int data_origin2 = (nnzA + nnzA + rowA) * sizeof(MAT_VAL_TYPE) + nnzA * sizeof(int) + (rowA + 1) * sizeof(MAT_PTR_TYPE);

    cusparse_spmv_all(csrValA, csrRowPtrA, csrColIdxA, X_val, dY_val, rowA, colA, nnzA, data_origin1, data_origin2, &pre_time, &cu_time, &cu_gflops, &cu_bandwidth1, &cu_bandwidth2);

    spmv_all(filename, csrValA, csrRowPtrA, csrColIdxA, X_val, Y_val, new_order, rowA, colA, nnzA, NUM, threshold, block_longest);

    FILE* fout;
    fout = fopen("data/spmv_f16_record.csv", "a");
    fprintf(fout, "%lld,%lf,%lf,%lf,%lf,%lf\n", data_origin1, pre_time, cu_time, cu_gflops, cu_bandwidth1, cu_bandwidth2);
    fclose(fout);

    // int result = verify_new(dY_val, Y_val, new_order, rowA);

    free(X_val);
    free(Y_val);
    free(dY_val);
    free(csrColIdxA);
    free(csrRowPtrA);
    free(csrValA);
    free(new_order);

    return 0;
}