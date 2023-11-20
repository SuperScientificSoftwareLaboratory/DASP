/*
    第一版整合版本，统一一个线程块4个warp
    long part： 一个warp进行两次mma，多个block得到一个y
    row_block part：
        根据row-block行数进行分情况讨论
        row-block < 59990                        一个warp算1个行块
        row-block >= 59990 && row-block < 400000 一个warp算2个行块
        row-block >= 400000                      一个warp算4个行块
        
    short part：一个warp进行四次mma，一个warp得到32个y
*/

#include "common.h"
#include "utils.h"

#define groupNum 1
#define warpNum_short 4
#define loopNum_short 4
#define warpNum_long 4
#define loopNum_long 2

__device__ __forceinline__ MAT_VAL_TYPE warpReduceSum(MAT_VAL_TYPE sum){
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

__device__ __forceinline__ void mma_m8n8k4_fp16(half *acc, half *frag_a, half *frag_b)
{
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __forceinline__ void mma_m8n8k4_fp16_v2(half *acc, uint32_t *A, half *frag_b)
{
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}

__device__ __forceinline__ void mma_m8n8k4_fp16_v3(uint32_t *C, uint32_t *A, uint32_t *B)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]):
        "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
    ); 
}


__device__ __forceinline__ int load_int_from_global(const int* a)
{
    int r;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ uint32_t load_uint_from_global(const uint32_t* a)
{
    uint32_t r;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ half load_half_from_global(const half* a)
{
    ushort r;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(r) : "l"(a));
    half *r_half = reinterpret_cast<half *>(&r);
    return *r_half;
}

__device__ __forceinline__ void store_half_to_global(const half* a, half v)
{
    ushort *v_u = reinterpret_cast<ushort *>(&v);
    asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(a), "h"(*v_u));
}


template <int rowloop>  // this parameter must be 1 or 2 or 4
__global__ void dasp_spmv2(uint32_t *dX_val, uint32_t *dY_val,
                          uint32_t *dlongA_val, int *dlongA_cid, MAT_VAL_TYPE *dwarp_val, MAT_PTR_TYPE *dlongA_rpt, int row_long,
                          uint32_t *dregA_val, int *dregA_cid, MAT_PTR_TYPE *dblockA_ptr, int row_block, int blocknum, 
                          uint32_t *dirregA_val, int *dirregA_cid, MAT_PTR_TYPE *dirregA_rpt,
                          uint32_t *dshort_val, int *dshort_cid, int short_row_1, int common_13, int short_row_34, int short_row_2,
                          int offset_reg, int offset_short1, int offset_short13, int offset_short34, int offset_short22,
                          MAT_PTR_TYPE fill0_nnz_short13, MAT_PTR_TYPE fill0_nnz_short34, MAT_PTR_TYPE fill0_nnz_short22)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int laneid = 31 & tid;
    
    int row = laneid < 16 ? (laneid >> 2) * 8 + (3 & laneid) : ((laneid - 16) >> 2) * 8 + (3 & laneid) + 4;
    int idx = row * MMA_K;
    int idx_val = row * 2;

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;
    int new_id = (7 & laneid) < 4 ? (laneid >> 3) * 4 + (3 & laneid) : (laneid >> 3) * 4 + (3 & laneid) + 16;

    MAT_VAL_TYPE const *valX_half = reinterpret_cast<MAT_VAL_TYPE const *>(&dX_val[0]);
    MAT_VAL_TYPE *valY_half = reinterpret_cast<MAT_VAL_TYPE *>(&dY_val[0]);

    if (bid < offset_reg)
    {
        // long part
        int global_warpid = bid * warpNum_long + (tid >> 5);

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;
        
        fragC[target_idx] = 0.0;
        
        #pragma unroll
        for (int i = 0; i < loopNum_long; i++)
        {
            int offset_cid = (global_warpid * loopNum_long + i) * MMA_M * MMA_K * 4;
            int offset_val = offset_cid >> 1;
            
            uint32_t *curA_val = dlongA_val + offset_val;
            int *curA_cid = dlongA_cid + offset_cid;

            fragA[0] = load_uint_from_global(curA_val + idx_val);
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1);
            fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
            fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
        }
        res = fragC[target_idx];
        res = warpReduceSum(res);

        if (laneid == 0)  
            store_half_to_global(dwarp_val + global_warpid, res);

        if (global_warpid >= row_long) return;

        int offset_long = load_int_from_global(dlongA_rpt + global_warpid);
        MAT_VAL_TYPE *cur_temp_val = dwarp_val + offset_long;
        int len = load_int_from_global(dlongA_rpt + global_warpid + 1) - offset_long;

        MAT_VAL_TYPE thread_val = 0;
        for (int i = laneid; i < len; i += WARP_SIZE)
        {
            thread_val += load_half_from_global(cur_temp_val + i);
        }
        thread_val = warpReduceSum(thread_val);

        if (laneid == 0)
            store_half_to_global(valY_half + global_warpid, thread_val); 
    }
    else if (bid >= offset_reg && bid < offset_short1)
    {
        // row-block part
        int bid_reg = bid - offset_reg;
        int warp_local = tid >> 5;

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;

        MAT_VAL_TYPE *valA_irreg = reinterpret_cast<MAT_VAL_TYPE *>(&dirregA_val[0]);

        if (rowloop == 1)
        {
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 4 + warp_local;
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;

            for (int i = 0; i < blocklen; i += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + i) >> 1);
                int *curA_cid = dregA_cid + offset_A + i;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 

            int offset_y = block_idx * BlockSize + laneid;
            if (laneid < 8 && offset_y < row_block)
            {
                int offset_irreg = load_int_from_global(dirregA_rpt + offset_y);
                int offset_irreg1 = load_int_from_global(dirregA_rpt + offset_y + 1);
                for (int i = offset_irreg; i < offset_irreg1; i ++)
                {   
                    res += load_half_from_global(valA_irreg + i) * valX_half[load_int_from_global(dirregA_cid + i)];
                }
                store_half_to_global(valY_half + row_long + offset_y, res);
                // valY_half[row_long + offset_y] = res;
            }
        }

        if (rowloop == 2)
        {
            MAT_VAL_TYPE result;
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 8 + warp_local * 2;
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];   
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if (laneid < 8) result =  res;

            // i = 1
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];   
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 1) result = res;

            int cur_row = bid_reg * 8 * BlockSize + warp_local * 2 * BlockSize + laneid;
            if (laneid < 16 && cur_row < row_block)
            {
                int offset_irreg = load_int_from_global(dirregA_rpt + cur_row);
                int offset_irreg1 = load_int_from_global(dirregA_rpt + cur_row + 1);
                for (int i = offset_irreg; i < offset_irreg1; i ++)
                {
                    result += load_half_from_global(valA_irreg + i) * valX_half[load_int_from_global(dirregA_cid + i)];
                }
                store_half_to_global(valY_half + row_long + cur_row, result);
            }
        }

        if (rowloop == 4)
        {
            MAT_VAL_TYPE result;

            // i = 0
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 16 + warp_local * 4;
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)]; 
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if (laneid < 8) result =  res;

            // i = 1
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 1) result = res;

            // i = 2
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_up_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 2) result = res;

            // i = 3
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = load_int_from_global(dblockA_ptr + block_idx);
            blocklen = load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = load_uint_from_global(curA_val + idx_val); 
                fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
                fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
                fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
                fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
                fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)]; 
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_up_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 3) result = res;

            int cur_row = bid_reg * 16 * BlockSize + warp_local * 4 * BlockSize + laneid;
            if (cur_row < row_block)
            {
                int offset_irreg = load_int_from_global(dirregA_rpt + cur_row);
                int offset_irreg1 = load_int_from_global(dirregA_rpt + cur_row + 1);
                for (int i = offset_irreg; i < offset_irreg1; i ++)
                {
                    result += load_half_from_global(valA_irreg + i) * valX_half[load_int_from_global(dirregA_cid + i)];
                }
                store_half_to_global(valY_half + row_long + cur_row, result);
            }
        }
    }
    else if (bid >= offset_short1 && bid < offset_short13)
    // if (bid >= offset_short1 && bid < offset_short13)
    {
        // short part - 1 nnz/row
        int bid1 = bid - offset_short1;
        int tid1 = bid1 * blockDim.x + tid;
        uint32_t *cur_val = dshort_val + ((fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22) >> 1);
        int *cur_cid = dshort_cid + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
        if (tid1 >= short_row_1)
        {
            return;
        }
        MAT_VAL_TYPE *valA = reinterpret_cast<MAT_VAL_TYPE *>(&cur_val[tid1 >> 1]);
        int x_idx = load_int_from_global(cur_cid + tid1);
        MAT_VAL_TYPE temp_y = load_half_from_global(valA + (1 & tid1)) * valX_half[x_idx];
        store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + short_row_34 + short_row_2 + tid1, temp_y);
        // valY_half[row_long + row_block + common_13 * 2 + short_row_34 + short_row_2 + tid1] = valA[tid1 % 2] * valX_half[cur_cid[tid1]];
    }
    else if (bid >= offset_short13 && bid < offset_short34)
    // if (bid >= offset_short13 && bid < offset_short34)
    {
        // short part - block 1&3
        int warpid_local = tid >> 5;
        int bid13 = bid - offset_short13;

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;

        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            // compute for 1
            fragC[target_idx] = 0.0;

            int offset = ((bid13 * groupNum + i) * warpNum_short + warpid_local) * MMA_M * MMA_K * 4;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            fragA[0] = load_uint_from_global(curA_val + idx_val); 
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
            fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
            fragB[1] = 0, fragB[2] = 0, fragB[3] = 0;  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);

            int offset_y = ((bid13 * groupNum + i) * warpNum_short  + warpid_local) * WARP_SIZE * 2 + laneid;
            if (offset_y < common_13 * 2) 
                valY_half[row_long + row_block + offset_y] = res;
            
            // compute for 3
            fragC[target_idx] = 0.0;

            fragB[0] = 0;
            fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
            // fragB[1] = valX_half[curA_cid[idx + 1]], fragB[2] = valX_half[curA_cid[idx + 2]], fragB[3] = valX_half[curA_cid[idx + 3]];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            
            offset_y += WARP_SIZE;
            if (offset_y < common_13 * 2) 
                store_half_to_global(valY_half + row_long + row_block + offset_y, res);
                // valY_half[row_long + row_block + offset_y] = res;
            
        }
    }
    else if (bid >= offset_short34 && bid < offset_short22)
    // if (bid >= offset_short34 && bid < offset_short22)
    {
        // short part - block3 & block4
        int warpid_local = tid >> 5;
        int bid34 = bid - offset_short34;

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;

        #pragma unroll
        for (int j = 0; j < groupNum; j ++)
        {
            fragC[target_idx] = 0.0;

            int offset = fill0_nnz_short13 + ((bid34 * groupNum + j) * warpNum_short + warpid_local) * MMA_M * MMA_K * loopNum_short;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;
            
            fragA[0] = load_uint_from_global(curA_val + idx_val); 
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
            fragB[0] = valX_half[load_int_from_global(curA_cid + idx)];
            fragB[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            
            int offset_y = ((bid34 * groupNum + j) * warpNum_short + warpid_local) * WARP_SIZE + laneid;
            if (offset_y < short_row_34) 
                store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + offset_y, res);
        }
    }
    else
    {
        // short part - blocl 2&2
        int warpid_local = tid >> 5;
        int bid22 = bid - offset_short22;

        uint32_t fragA[2], fragB[2], fragC[4];
        MAT_VAL_TYPE res;
        
        MAT_VAL_TYPE *fragB_half = reinterpret_cast<MAT_VAL_TYPE *>(&fragB[0]);
        MAT_VAL_TYPE *fragC_half = reinterpret_cast<MAT_VAL_TYPE *>(&fragC[0]);
        
        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            // compute for 2 (1)
            fragC_half[target_idx] = 0.0;

            int offset_block = ((bid22 * groupNum + i) * warpNum_short + warpid_local) * WARP_SIZE;
            int offset = fill0_nnz_short13 + fill0_nnz_short34 + offset_block * 4;
            int offset_y = offset_block * 2 + laneid;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            // fragA[0] = curA_val[idx_val], fragA[1] = curA_val[idx_val + 1];
            fragA[0] = load_uint_from_global(curA_val + idx_val); 
            fragA[1] = load_uint_from_global(curA_val + idx_val + 1); 
            fragB_half[0] = valX_half[load_int_from_global(curA_cid + idx)]; 
            fragB_half[1] = valX_half[load_int_from_global(curA_cid + idx + 1)];
            fragB[1] = 0;  
            mma_m8n8k4_fp16_v3(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC_half[target_idx], new_id);

            if (offset_y < short_row_2) 
                store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + short_row_34 + offset_y, res);

                // valY_half[row_long + row_block + common_13 * 2 + short_row_34 + offset_y] = res;
            
            // compute for 2 (2)        
            fragC_half[target_idx] = 0.0;

            fragB[0] = 0;
            fragB_half[2] = valX_half[load_int_from_global(curA_cid + idx + 2)];
            fragB_half[3] = valX_half[load_int_from_global(curA_cid + idx + 3)];  
            mma_m8n8k4_fp16_v3(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC_half[target_idx], new_id);

            offset_y += WARP_SIZE;
            if (offset_y < short_row_2) 
                store_half_to_global(valY_half + row_long + row_block + common_13 * 2 + short_row_34 + offset_y, res);
                // valY_half[row_long + row_block + common_13 * 2 + short_row_34 + offset_y] = res;
        }
    }
}


template <int rowloop>  // this parameter must be 1 or 2 or 4
__global__ void dasp_spmv(uint32_t *dX_val, uint32_t *dY_val,
                          uint32_t *dlongA_val, int *dlongA_cid, MAT_VAL_TYPE *dwarp_val, MAT_PTR_TYPE *dlongA_rpt, int row_long,
                          uint32_t *dregA_val, int *dregA_cid, MAT_PTR_TYPE *dblockA_ptr, int row_block, int blocknum, 
                          uint32_t *dirregA_val, int *dirregA_cid, MAT_PTR_TYPE *dirregA_rpt,
                          uint32_t *dshort_val, int *dshort_cid, int short_row_1, int common_13, int short_row_34, int short_row_2,
                          int offset_reg, int offset_short1, int offset_short13, int offset_short34, int offset_short22,
                          MAT_PTR_TYPE fill0_nnz_short13, MAT_PTR_TYPE fill0_nnz_short34, MAT_PTR_TYPE fill0_nnz_short22)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int laneid = 31 & tid;
    
    int row = laneid < 16 ? (laneid >> 2) * 8 + (3 & laneid) : ((laneid - 16) >> 2) * 8 + (3 & laneid) + 4;
    int idx = row * MMA_K;
    int idx_val = row * 2;

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;
    int new_id = (7 & laneid) < 4 ? (laneid >> 3) * 4 + (3 & laneid) : (laneid >> 3) * 4 + (3 & laneid) + 16;

    MAT_VAL_TYPE const *valX_half = reinterpret_cast<MAT_VAL_TYPE const *>(&dX_val[0]);
    MAT_VAL_TYPE *valY_half = reinterpret_cast<MAT_VAL_TYPE *>(&dY_val[0]);

    if (bid < offset_reg)
    {
        // long part
        int global_warpid = bid * warpNum_long + (tid >> 5);

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;
        
        fragC[target_idx] = 0.0;
        
        #pragma unroll
        for (int i = 0; i < loopNum_long; i++)
        {
            int offset_cid = (global_warpid * loopNum_long + i) * MMA_M * MMA_K * 4;
            int offset_val = offset_cid >> 1;
            
            uint32_t *curA_val = dlongA_val + offset_val;
            int *curA_cid = dlongA_cid + offset_cid;

            fragA[0] = curA_val[idx_val], fragA[1] = curA_val[idx_val + 1];
            fragB[0] = valX_half[curA_cid[idx]], fragB[1] = valX_half[curA_cid[idx + 1]], fragB[2] = valX_half[curA_cid[idx + 2]], fragB[3] = valX_half[curA_cid[idx + 3]];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
        }
        res = fragC[target_idx];
        res = warpReduceSum(res);

        if (laneid == 0) dwarp_val[global_warpid] = res;

        if (global_warpid >= row_long) return;

        MAT_VAL_TYPE *cur_temp_val = dwarp_val + dlongA_rpt[global_warpid];
        int len = dlongA_rpt[global_warpid + 1] - dlongA_rpt[global_warpid];

        MAT_VAL_TYPE thread_val = 0;
        for (int i = laneid; i < len; i += WARP_SIZE)
        {
            thread_val += cur_temp_val[i];
        }
        thread_val = warpReduceSum(thread_val);

        if (laneid == 0) valY_half[global_warpid] = thread_val;
    }
    else if (bid >= offset_reg && bid < offset_short1)
    {
        // row-block part
        int bid_reg = bid - offset_reg;
        int warp_local = tid >> 5;

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;

        MAT_VAL_TYPE *valA_irreg = reinterpret_cast<MAT_VAL_TYPE *>(&dirregA_val[0]);

        if (rowloop == 1)
        {
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 4 + warp_local;
            int offset_A = dblockA_ptr[block_idx];
            int blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            // if (block_idx >= blocknum) return;

            for (int i = 0; i < blocklen; i += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + i) >> 1);
                int *curA_cid = dregA_cid + offset_A + i;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 

            int offset_y = block_idx * BlockSize + laneid;
            if (laneid < 8 && offset_y < row_block)
            {
                for (int i = dirregA_rpt[offset_y]; i < dirregA_rpt[offset_y + 1]; i ++)
                {
                    res += valA_irreg[i] * valX_half[dirregA_cid[i]];
                }
                valY_half[row_long + offset_y] = res;
            }
        }

        if (rowloop == 2)
        {
            MAT_VAL_TYPE result;
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 8 + warp_local * 2;
            int offset_A = dblockA_ptr[block_idx];
            int blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if (laneid < 8) result =  res;

            // i = 1
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = dblockA_ptr[block_idx];
            blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 1) result = res;

            int cur_row = bid_reg * 8 * BlockSize + warp_local * 2 * BlockSize + laneid;
            if (laneid < 16 && cur_row < row_block)
            {
                for (int i = dirregA_rpt[cur_row]; i < dirregA_rpt[cur_row + 1]; i ++)
                {
                    result += valA_irreg[i] * valX_half[dirregA_cid[i]];
                }
                valY_half[row_long + cur_row] = result;
            }
        }

        if (rowloop == 4)
        {
            MAT_VAL_TYPE result;

            // i = 0
            fragC[target_idx] = 0.0;

            int block_idx = bid_reg * 16 + warp_local * 4;
            int offset_A = dblockA_ptr[block_idx];
            int blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if (laneid < 8) result =  res;

            // i = 1
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = dblockA_ptr[block_idx];
            blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_down_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 1) result = res;

            // i = 2
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = dblockA_ptr[block_idx];
            blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_up_sync(0xffffffff, res, 16);
            res += __shfl_down_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 2) result = res;

            // i = 3
            fragC[target_idx] = 0.0;

            block_idx += 1;
            offset_A = dblockA_ptr[block_idx];
            blocklen = dblockA_ptr[block_idx + 1] - offset_A;
            
            for (int j = 0; j < blocklen; j += (MMA_M * MMA_K * 4))
            {
                uint32_t *curA_val = dregA_val + ((offset_A + j) >> 1);
                int *curA_cid = dregA_cid + offset_A + j;

                fragA[0] = curA_val[idx_val];
                fragA[1] = curA_val[idx_val + 1];
                fragB[0] = valX_half[curA_cid[idx]];
                fragB[1] = valX_half[curA_cid[idx + 1]];
                fragB[2] = valX_half[curA_cid[idx + 2]];
                fragB[3] = valX_half[curA_cid[idx + 3]];  
                mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            }
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            res += __shfl_up_sync(0xffffffff, res, 16);
            res += __shfl_up_sync(0xffffffff, res, 8); 
            if ((laneid >> 3) == 3) result = res;

            int cur_row = bid_reg * 16 * BlockSize + warp_local * 4 * BlockSize + laneid;
            if (cur_row < row_block)
            {
                for (int i = dirregA_rpt[cur_row]; i < dirregA_rpt[cur_row + 1]; i ++)
                {
                    result += valA_irreg[i] * valX_half[dirregA_cid[i]];
                }
                valY_half[row_long + cur_row] = result;
            }
        }
    }
    else if (bid >= offset_short1 && bid < offset_short13)
    {
        // short part - 1 nnz/row
        int bid1 = bid - offset_short1;
        int tid1 = bid1 * blockDim.x + tid;
        uint32_t *cur_val = dshort_val + ((fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22) >> 1);
        int *cur_cid = dshort_cid + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
        if (tid1 >= short_row_1)
        {
            return;
        }
        MAT_VAL_TYPE *valA = reinterpret_cast<MAT_VAL_TYPE *>(&cur_val[tid1 >> 1]);
        valY_half[row_long + row_block + common_13 * 2 + short_row_34 + short_row_2 + tid1] = valA[tid1 % 2] * valX_half[cur_cid[tid1]];
    }
    else if (bid >= offset_short13 && bid < offset_short34)
    {
        // short part - block 1&3
        int warpid_local = tid >> 5;
        int bid13 = bid - offset_short13;

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;

        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            // compute for 1
            fragC[target_idx] = 0.0;

            int offset = ((bid13 * groupNum + i) * warpNum_short + warpid_local) * MMA_M * MMA_K * 4;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            fragA[0] = curA_val[idx_val], fragA[1] = curA_val[idx_val + 1];
            fragB[0] = valX_half[curA_cid[idx]], fragB[1] = 0, fragB[2] = 0, fragB[3] = 0;  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);

            int offset_y = ((bid13 * groupNum + i) * warpNum_short  + warpid_local) * WARP_SIZE * 2 + laneid;
            if (offset_y < common_13 * 2) 
                valY_half[row_long + row_block + offset_y] = res;
            
            // compute for 3
            fragC[target_idx] = 0.0;

            fragB[0] = 0, fragB[1] = valX_half[curA_cid[idx + 1]], fragB[2] = valX_half[curA_cid[idx + 2]], fragB[3] = valX_half[curA_cid[idx + 3]];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            
            offset_y += WARP_SIZE;
            if (offset_y < common_13 * 2) 
                valY_half[row_long + row_block + offset_y] = res;
            
        }
    }
    else if (bid >= offset_short34 && bid < offset_short22)
    {
        // short part - block3 & block4
        int warpid_local = tid >> 5;
        int bid34 = bid - offset_short34;

        uint32_t fragA[2];
        MAT_VAL_TYPE fragB[4], fragC[8], res;

        #pragma unroll
        for (int j = 0; j < groupNum; j ++)
        {
            fragC[target_idx] = 0.0;

            int offset = fill0_nnz_short13 + ((bid34 * groupNum + j) * warpNum_short + warpid_local) * MMA_M * MMA_K * loopNum_short;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            fragA[0] = curA_val[idx_val], fragA[1] = curA_val[idx_val + 1];
            fragB[0] = valX_half[curA_cid[idx]], fragB[1] = valX_half[curA_cid[idx + 1]], fragB[2] = valX_half[curA_cid[idx + 2]], fragB[3] = valX_half[curA_cid[idx + 3]];  
            mma_m8n8k4_fp16_v2(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC[target_idx], new_id);
            
            int offset_y = ((bid34 * groupNum + j) * warpNum_short + warpid_local) * WARP_SIZE + laneid;
            if (offset_y < short_row_34) 
                valY_half[row_long + row_block + common_13 * 2 + offset_y] = res;
        }
    }
    else
    {
        // short part - blocl 2&2
        int warpid_local = tid >> 5;
        int bid22 = bid - offset_short22;

        uint32_t fragA[2], fragB[2], fragC[4];
        MAT_VAL_TYPE res;
        
        MAT_VAL_TYPE *fragB_half = reinterpret_cast<MAT_VAL_TYPE *>(&fragB[0]);
        MAT_VAL_TYPE *fragC_half = reinterpret_cast<MAT_VAL_TYPE *>(&fragC[0]);
        
        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            // compute for 2 (1)
            fragC_half[target_idx] = 0.0;

            int offset_block = ((bid22 * groupNum + i) * warpNum_short + warpid_local) * WARP_SIZE;
            int offset = fill0_nnz_short13 + fill0_nnz_short34 + offset_block * 4;
            int offset_y = offset_block * 2 + laneid;
            uint32_t *curA_val = dshort_val + (offset >> 1);
            int *curA_cid = dshort_cid + offset;

            fragA[0] = curA_val[idx_val], fragA[1] = curA_val[idx_val + 1];
            fragB_half[0] = valX_half[curA_cid[idx]], fragB_half[1] = valX_half[curA_cid[idx + 1]];
            fragB[1] = 0;  
            mma_m8n8k4_fp16_v3(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC_half[target_idx], new_id);

            if (offset_y < short_row_2) 
                valY_half[row_long + row_block + common_13 * 2 + short_row_34 + offset_y] = res;
            
            // compute for 2 (2)        
            fragC_half[target_idx] = 0.0;

            fragB[0] = 0;
            fragB_half[2] = valX_half[curA_cid[idx + 2]], fragB_half[3] = valX_half[curA_cid[idx + 3]];  
            mma_m8n8k4_fp16_v3(fragC, fragA, fragB);
            res = __shfl_sync(0xffffffff, fragC_half[target_idx], new_id);

            offset_y += WARP_SIZE;
            if (offset_y < short_row_2) 
                valY_half[row_long + row_block + common_13 * 2 + short_row_34 + offset_y] = res;
        }
    }
}

__host__ void spmv_all(char *filename, MAT_VAL_TYPE *csrValA, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, 
                      MAT_VAL_TYPE *X_val, MAT_VAL_TYPE *Y_val, int *order_rid, int rowA, int colA, MAT_PTR_TYPE nnzA, int NUM, double threshold, int block_longest)
{
    struct timeval t1, t2, t3, pre_t1, pre_t2;

    // three parts: short row (1 & 3 & 2 & 4), long row, row-block (regular（origin & fill0） & irregular)
    gettimeofday(&pre_t1, NULL);
    MAT_PTR_TYPE nnz_short, nnz_long, origin_nnz_reg, fill0_nnz_reg, nnz_irreg;
    int row_long = 0, row_block = 0, row_zero = 0;

    // get the short part data
    // (short_val, short_cid)
    int short_row_1 = 0, short_row_3 = 0, short_row_2 = 0, short_row_4 = 0;

    for (int i = 0; i < rowA; i ++)
    {
        int row_len = csrRowPtrA[i + 1] - csrRowPtrA[i];
        if (row_len == 1)
        {   
            short_row_1 ++;
        }
        else if (row_len == 3)
        {
            short_row_3 ++;
        }
        else if (row_len == 2)
        {
            short_row_2 ++;
        }
        else if (row_len == 0)
        {
            row_zero ++;
        }
        else if (row_len == 4)
        {
            short_row_4 ++;
        }
        // else if (row_len >= warpNum_long * loopNum_long * MMA_M * MMA_K)
        else if (row_len >= block_longest)
        {
            row_long ++;
        }
        else
        {
            row_block ++;
        }
    }

    int rowloop;
    if (row_block < 59990) rowloop = 1;
    else if (row_block >= 59990 && row_block < 400000) rowloop = 2;
    else rowloop = 4;

    int *short_rid_1 = (int *)malloc(sizeof(int) * short_row_1);
    int *short_rid_2 = (int *)malloc(sizeof(int) * short_row_2);
    int *short_rid_3 = (int *)malloc(sizeof(int) * short_row_3);
    int *short_rid_4 = (int *)malloc(sizeof(int) * short_row_4);
    int *long_rid = (int *)malloc(sizeof(int) * row_long);
    int *zero_rid = (int *)malloc(sizeof(int) * row_zero);
    int *ridA = (int *)malloc(sizeof(int) * row_block);

    MAT_PTR_TYPE *rptA = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row_block + 1));
    memset(rptA, 0, sizeof(MAT_PTR_TYPE) * (row_block + 1));
    MAT_PTR_TYPE *long_rpt = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row_long + 1));
    memset(long_rpt, 0, sizeof(MAT_PTR_TYPE) * (row_long + 1));

    int short_row_flag1 = 0, short_row_flag3 = 0, short_row_flag2 = 0, short_row_flag4 = 0;
    int row_long_flag = 0, flag0 = 0, row_block_flag = 0;
    for (int i = 0; i < rowA; i ++)
    {
        int row_len = csrRowPtrA[i + 1] - csrRowPtrA[i];
        if (row_len == 1)
        {
            short_rid_1[short_row_flag1] = i;
            short_row_flag1 ++;
        }
        else if (row_len == 3)
        {
            short_rid_3[short_row_flag3] = i;
            short_row_flag3 ++;
        }
        else if (row_len == 2)
        {
            short_rid_2[short_row_flag2] = i;
            short_row_flag2 ++;
        }
        else if (row_len == 0)
        {
            zero_rid[flag0] = i;
            flag0 ++;
        }
        else if (row_len == 4)
        {
            short_rid_4[short_row_flag4] = i;
            short_row_flag4 ++;
        }
        // else if (row_len >= warpNum_long * loopNum_long * MMA_M * MMA_K)
        else if (row_len >= block_longest)
        {
            long_rpt[row_long_flag] = row_len;
            long_rid[row_long_flag] = i;
            row_long_flag ++;
        }
        else
        {
            rptA[row_block_flag] = row_len;
            ridA[row_block_flag] = i;
            row_block_flag ++;
        }
    } 
    nnz_short = short_row_1 + short_row_3 * 3 + short_row_2 * 2 + short_row_4 * 4;
 
    int common_13 = short_row_1 < short_row_3 ? short_row_1 : short_row_3;
    if (common_13 / BlockSize >= 16)
    {
        common_13 = BlockSize * 4 * (common_13 / (BlockSize * 4));
        short_row_1 = short_row_1 - common_13;
        short_row_3 = short_row_3 - common_13;
    }
    else
    {
        common_13 = 0;
    }

    int short_block13 = (common_13 + BlockSize - 1) / BlockSize;  
    int half_short_row_2 = (short_row_2 + 1) / 2;
    int short_block22 = (half_short_row_2 + BlockSize - 1) / BlockSize;
    int short_row_34 = short_row_3 + short_row_4;
    int short_block34 = (short_row_34 + BlockSize - 1) / BlockSize;

    int block13_per_threadblock = warpNum_short * groupNum * 4;
    int block22_per_threadblock = warpNum_short * groupNum * 4;
    int block34_per_threadblock = warpNum_short * groupNum * loopNum_short;

    int threadblock13 = (short_block13 + block13_per_threadblock - 1) / block13_per_threadblock;
    int threadblock22 = (short_block22 + block22_per_threadblock - 1) / block22_per_threadblock;
    int threadblock34 = (short_block34 + block34_per_threadblock - 1) / block34_per_threadblock;

    MAT_PTR_TYPE fill0_nnz_short13 = threadblock13 * block13_per_threadblock * MMA_M * MMA_K;
    MAT_PTR_TYPE fill0_nnz_short34 = threadblock34 * block34_per_threadblock * MMA_M * MMA_K;
    MAT_PTR_TYPE fill0_nnz_short22 = threadblock22 * block22_per_threadblock * MMA_M * MMA_K;
    MAT_PTR_TYPE fill0_nnz_short = ((short_row_1 + 1) / 2) * 2 + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
    MAT_VAL_TYPE *short_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * fill0_nnz_short);
    int *short_cid = (int *)malloc(sizeof(int) * fill0_nnz_short);
    memset(short_val, 0.0, sizeof(MAT_VAL_TYPE) * fill0_nnz_short);
    memset(short_cid, 0, sizeof(int) * fill0_nnz_short);

    #pragma omp parallel for
    for (int i = 0; i < short_block13; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + i * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + i * MMA_M * MMA_K;

        for (int j = 0; j < BlockSize && i * BlockSize + j < common_13; j ++)
        {
            int cur_row_1 = short_rid_1[short_row_1 + i * BlockSize + j];
            int cur_row_3 = short_rid_3[i * BlockSize + j];
            cur_short_val[j * MMA_K] = csrValA[csrRowPtrA[cur_row_1]];
            cur_short_cid[j * MMA_K] = csrColIdxA[csrRowPtrA[cur_row_1]];
            cur_short_val[j * MMA_K + 1] = csrValA[csrRowPtrA[cur_row_3]];
            cur_short_val[j * MMA_K + 2] = csrValA[csrRowPtrA[cur_row_3] + 1];
            cur_short_val[j * MMA_K + 3] = csrValA[csrRowPtrA[cur_row_3] + 2];
            cur_short_cid[j * MMA_K + 1] = csrColIdxA[csrRowPtrA[cur_row_3]];
            cur_short_cid[j * MMA_K + 2] = csrColIdxA[csrRowPtrA[cur_row_3] + 1];
            cur_short_cid[j * MMA_K + 3] = csrColIdxA[csrRowPtrA[cur_row_3] + 2];
        }
    }   

    #pragma omp parallel for
    for (int i = 0; i < short_row_3; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + fill0_nnz_short13 + i * MMA_K;
        int *cur_short_cid = short_cid + fill0_nnz_short13 + i * MMA_K;
        
        int cur_row = short_rid_3[common_13 + i];

        cur_short_val[0] = csrValA[csrRowPtrA[cur_row]];
        cur_short_val[1] = csrValA[csrRowPtrA[cur_row] + 1]; 
        cur_short_val[2] = csrValA[csrRowPtrA[cur_row] + 2]; 
        cur_short_cid[0] = csrColIdxA[csrRowPtrA[cur_row]];
        cur_short_cid[1] = csrColIdxA[csrRowPtrA[cur_row] + 1]; 
        cur_short_cid[2] = csrColIdxA[csrRowPtrA[cur_row] + 2]; 
    }

    #pragma omp parallel for
    for (int i = 0; i < short_row_4; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + fill0_nnz_short13 + (short_row_3 + i) * MMA_K;
        int *cur_short_cid = short_cid + fill0_nnz_short13 + (short_row_3 + i) * MMA_K;
        
        int cur_row = short_rid_4[i];

        cur_short_val[0] = csrValA[csrRowPtrA[cur_row]];
        cur_short_val[1] = csrValA[csrRowPtrA[cur_row] + 1]; 
        cur_short_val[2] = csrValA[csrRowPtrA[cur_row] + 2]; 
        cur_short_val[3] = csrValA[csrRowPtrA[cur_row] + 3]; 
        cur_short_cid[0] = csrColIdxA[csrRowPtrA[cur_row]];
        cur_short_cid[1] = csrColIdxA[csrRowPtrA[cur_row] + 1]; 
        cur_short_cid[2] = csrColIdxA[csrRowPtrA[cur_row] + 2]; 
        cur_short_cid[3] = csrColIdxA[csrRowPtrA[cur_row] + 3]; 
    }

    int group22 = (short_block22 + 3) / 4;
    #pragma omp parallel for
    for (int i = 0; i < group22; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + fill0_nnz_short13 + fill0_nnz_short34 + i * 4 * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + fill0_nnz_short13 + fill0_nnz_short34 + i * 4 * MMA_M * MMA_K;

        for (int j = 0; j < (BlockSize * 4 * 2) && (i * BlockSize * 4 * 2 + j) < short_row_2; j ++)
        {
            int cur_row = short_rid_2[i * BlockSize * 4 * 2 + j];
            cur_short_val[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2] = csrValA[csrRowPtrA[cur_row]];
            cur_short_val[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2 + 1] = csrValA[csrRowPtrA[cur_row] + 1];
            cur_short_cid[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2] = csrColIdxA[csrRowPtrA[cur_row]];
            cur_short_cid[(j % (BlockSize * 4)) * MMA_K + (j / (BlockSize * 4)) * 2 + 1] = csrColIdxA[csrRowPtrA[cur_row] + 1];
        }
    }
    
    int offset_short_row1 = fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
    #pragma omp parallel for
    for (int i = 0; i < short_row_1; i ++)
    {
        int cur_row = short_rid_1[i];
        short_val[offset_short_row1 + i] = csrValA[csrRowPtrA[cur_row]];
        short_cid[offset_short_row1 + i] = csrColIdxA[csrRowPtrA[cur_row]];
    }

    // resort except rows
    radix_sort(rptA, ridA, row_block);

    // get the data except short row part
    // (rptA, cidA, valA)
    exclusive_scan(rptA, row_block + 1);
    exclusive_scan(long_rpt, row_long + 1);
    nnz_long = long_rpt[row_long];

    //record the sort order
    memcpy(order_rid, long_rid, sizeof(int) * row_long);
    memcpy(order_rid + row_long, ridA, sizeof(int) * row_block);
    int group13 = common_13 / (4 * BlockSize);
    #pragma omp parallel for
    for (int i = 0; i < group13; i ++)
    {
        int *cur_order_rid = order_rid + row_long + row_block + i * BlockSize * 4 * 2;
        for (int j = 0; j < BlockSize * 4; j ++)
        {
            cur_order_rid[j] = short_rid_1[short_row_1 + i * BlockSize * 4 + j];
            cur_order_rid[BlockSize * 4 + j] = short_rid_3[i * BlockSize * 4 + j];
        }
    }
    memcpy(order_rid + row_long + row_block + common_13 * 2, short_rid_3 + common_13, sizeof(int) * short_row_3);
    memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3, short_rid_4, sizeof(int) * short_row_4);
    memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3 + short_row_4, short_rid_2, sizeof(int) * short_row_2);
    memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3 + short_row_4 + short_row_2, short_rid_1, sizeof(int) * short_row_1);
    memcpy(order_rid + row_long + row_block + common_13 * 2 + short_row_3 + short_row_4 + short_row_2 + short_row_1, zero_rid, sizeof(int) * row_zero);

    // get the long part data
    MAT_PTR_TYPE *long_rpt_new = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row_long + 1));
    memset(long_rpt_new, 0, sizeof(MAT_PTR_TYPE) * (row_long + 1));
    int warp_number = 0;
    #pragma omp parallel for
    for (int i = 0; i < row_long; i ++)
    {
        int nnz_num = long_rpt[i + 1] - long_rpt[i];
        int cur_warp_num = (nnz_num + MMA_M * MMA_K * loopNum_long * 4 - 1) / (MMA_M * MMA_K * loopNum_long * 4);
        long_rpt_new[i] = cur_warp_num;
    }
    exclusive_scan(long_rpt_new, row_long + 1);
    warp_number = long_rpt_new[row_long];

    int BlockNum_long = (warp_number + warpNum_long - 1) / warpNum_long;
    int fill0_nnz_long = BlockNum_long * warpNum_long * loopNum_long * 4 * MMA_M * MMA_K;
    warp_number = BlockNum_long * warpNum_long;
    MAT_VAL_TYPE *val_by_warp = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * warp_number);
    int *rid_by_warp = (int *)malloc(sizeof(int) * warp_number);
    MAT_VAL_TYPE *long_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * fill0_nnz_long);
    memset(long_val, 0.0, sizeof(MAT_VAL_TYPE) * fill0_nnz_long);
    int *long_cid = (int *)malloc(sizeof(int) * fill0_nnz_long);
    memset(long_cid, 0, sizeof(int) * fill0_nnz_long);

    #pragma omp parallel for
    for (int i = 0; i < row_long; i ++)
    {
        MAT_VAL_TYPE *cur_val = long_val + long_rpt_new[i] * loopNum_long * 4 * MMA_M * MMA_K;
        int *cur_cid = long_cid + long_rpt_new[i] * loopNum_long * 4 * MMA_M * MMA_K;
        int real_rid = long_rid[i];
        if (csrRowPtrA[real_rid + 1] - csrRowPtrA[real_rid] != long_rpt[i + 1] - long_rpt[i]) printf("error!\n");
        
        for (int j = 0; j < long_rpt[i + 1] - long_rpt[i]; j ++)
        {
            cur_val[j] = csrValA[csrRowPtrA[real_rid] + j];
            cur_cid[j] = csrColIdxA[csrRowPtrA[real_rid] + j];
        }

        for (int j = long_rpt_new[i]; j < long_rpt_new[i + 1]; j ++)
        {
            rid_by_warp[j] = i;
        }
    }

    // preprocessing the row-block part : divide that into regular part and irregular part  
    int blocknum = (row_block + BlockSize - 1) / BlockSize;
    blocknum = ((blocknum + rowloop * 4 - 1) / (rowloop * 4)) * rowloop * 4;
    MAT_PTR_TYPE *blockPtr = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (blocknum + 1));
    memset(blockPtr, 0, sizeof(MAT_PTR_TYPE) * (blocknum + 1));

    MAT_PTR_TYPE *irreg_rpt = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row_block + 1));
    memset(irreg_rpt, 0, sizeof(MAT_PTR_TYPE) * (row_block + 1));

    #pragma omp parallel for
    for (int i = 0; i < blocknum; i++)
    {   
        int row_start = i * BlockSize;
        int row_end = (i + 1) * BlockSize >= row_block ? row_block : (i + 1) * BlockSize;
        int k = 1;
        while(1)
        {
            int block_nnz = 0;
            for (int cur_row = row_start; cur_row < row_end; cur_row++)
            {
                int row_len = rptA[cur_row + 1] - rptA[cur_row];
                if (row_len / MMA_K >= k) block_nnz += MMA_K;
                else if(row_len / MMA_K == k - 1) block_nnz += row_len % MMA_K;
            }
            
            if (block_nnz >= threshold * MMA_K * MMA_M)
            {
                blockPtr[i] += MMA_K * MMA_M;
            }
            else
            {
                for (int cur_row = row_start; cur_row < row_end; cur_row++ )
                {
                    int row_len = rptA[cur_row + 1] - rptA[cur_row];
                    irreg_rpt[cur_row] = row_len - (k - 1) * MMA_K > 0 ? row_len - (k - 1) * MMA_K : 0;
                }
                break;
            }
            k++;
        }
        blockPtr[i] = ((blockPtr[i] + MMA_M * MMA_K * 4 - 1) / (MMA_M * MMA_K * 4)) * (MMA_M * MMA_K * 4);
    }
    
    exclusive_scan(blockPtr, blocknum + 1);
    exclusive_scan(irreg_rpt, row_block + 1);
    
    // int offset_row_block = row_long;
    fill0_nnz_reg = blockPtr[blocknum];
    nnz_irreg = irreg_rpt[row_block];
    origin_nnz_reg = nnzA - nnz_irreg - nnz_long - nnz_short;

    // get the row-block part data---irregular part
    MAT_PTR_TYPE fill0_nnz_irreg = ((nnz_irreg + 1) / 2) * 2;
    MAT_VAL_TYPE *irreg_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * fill0_nnz_irreg);
    int *irreg_cid = (int *)malloc(sizeof(int) * nnz_irreg);
    #pragma omp parallel for
    for (int i = 0; i < row_block; i ++)
    {
        int cur_rid = ridA[i];
        int irreg_offset = irreg_rpt[i];
        int irreg_len = irreg_rpt[i + 1] - irreg_offset;
        for (int j = 0; j < irreg_len; j ++)
        {
            irreg_val[irreg_offset + j] = csrValA[csrRowPtrA[cur_rid + 1] - irreg_len + j];
            irreg_cid[irreg_offset + j] = csrColIdxA[csrRowPtrA[cur_rid + 1] - irreg_len + j];
        }
    }

    // get the row_block part data---regular part
    MAT_VAL_TYPE *reg_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * fill0_nnz_reg);
    int *reg_cid = (int *)malloc(sizeof(int) * fill0_nnz_reg);
    
    #pragma omp parallel for
    for (int bid = 0; bid < blocknum; bid ++)
    {
        int nnz_block = (blockPtr[bid + 1] - blockPtr[bid]);
        int blocklen = nnz_block / BlockSize;

        for (int rowid = bid * BlockSize; rowid < (bid + 1) * BlockSize; rowid ++)
        {
            int regA_start = blockPtr[bid] + blocklen * (rowid - bid * BlockSize);
            if (rowid < row_block)
            {
                int real_id = ridA[rowid];
                int A_start = csrRowPtrA[real_id];
                // int row_len = csrRowPtrA[real_id + 1] - A_start;
                int row_len = csrRowPtrA[real_id + 1] - A_start - (irreg_rpt[rowid + 1] - irreg_rpt[rowid]);
                for (int i = 0; i < blocklen; i ++)
                {
                    if (i < row_len)
                    {
                        reg_val[regA_start + i] = csrValA[A_start + i];
                        reg_cid[regA_start + i] = csrColIdxA[A_start + i];
                    }
                    else
                    {
                        reg_val[regA_start + i] = 0;
                        reg_cid[regA_start + i] = 0;
                    }
                }
            }
            else
            {
                for (int i = 0; i < blocklen; i ++)
                {
                    reg_val[regA_start + i] = 0.0;
                    reg_cid[regA_start + i] = 0;
                }
            }

        }

        MAT_VAL_TYPE *temp_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * nnz_block);
        int *temp_cid = (int *)malloc(sizeof(int) * nnz_block);
        MAT_VAL_TYPE *cur_val = reg_val + blockPtr[bid];
        int *cur_cid = reg_cid + blockPtr[bid];

        for (int i = 0; i < nnz_block; i ++)
        {
            int new_id = ((i % blocklen) / MMA_K) * BlockSize * MMA_K + (i / blocklen) * MMA_K + i % MMA_K;
            temp_val[new_id] = cur_val[i];
            temp_cid[new_id] = cur_cid[i];
        }
        memcpy(cur_val, temp_val, sizeof(MAT_VAL_TYPE) * nnz_block);
        memcpy(cur_cid, temp_cid, sizeof(int) * nnz_block);
        free(temp_val);
        free(temp_cid);
    }
    gettimeofday(&pre_t2, NULL);
    double dasp_pre = (pre_t2.tv_sec - pre_t1.tv_sec) * 1000.0 + (pre_t2.tv_usec - pre_t1.tv_usec) / 1000.0;
    // printf("dasp preprocessing time: %8.4lf ms\n", dasp_pre);

    long fill0_nnz = fill0_nnz_short + fill0_nnz_long + nnz_irreg + fill0_nnz_reg;
    double rate_fill0 = (double)(fill0_nnz - nnzA) / nnzA;
    
    long long int data_X = (rowA + colA) * sizeof(MAT_VAL_TYPE) + \
                           fill0_nnz_long * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + warp_number * sizeof(MAT_VAL_TYPE) + (row_long + 1) * sizeof(int) + \
                           fill0_nnz_short * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + \
                           fill0_nnz_reg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (blocknum + 1) * sizeof(MAT_PTR_TYPE) + \
                           fill0_nnz_irreg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (row_block + 1) * sizeof(MAT_PTR_TYPE);
    
    long long int data_X2 = (rowA + nnzA) * sizeof(MAT_VAL_TYPE) + \
                            fill0_nnz_long * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + warp_number * sizeof(MAT_VAL_TYPE) + (row_long + 1) * sizeof(int) + \
                            fill0_nnz_short * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + \
                            fill0_nnz_reg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (blocknum + 1) * sizeof(MAT_PTR_TYPE) + \
                            fill0_nnz_irreg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (row_block + 1) * sizeof(MAT_PTR_TYPE);
    
    int BlockNum = (blocknum + rowloop * 4 - 1) / (rowloop * 4);

    int ThreadNum_short = warpNum_short * WARP_SIZE;
    int BlockNum_short_1 = (short_row_1 + ThreadNum_short - 1) / ThreadNum_short;
    int BlockNum_short = BlockNum_short_1 + threadblock13 + threadblock34 + threadblock22;

    int offset_reg = BlockNum_long;
    int offset_short1 = offset_reg + BlockNum;
    int offset_short13 = offset_short1 + BlockNum_short_1;
    int offset_short34 = offset_short13 + threadblock13;
    int offset_short22 = offset_short34 + threadblock34;

    int BlockNum_all = BlockNum_long + BlockNum + BlockNum_short;
    int ThreadNum_all = 4 * WARP_SIZE;
    
    uint32_t *dX_val, *dY_val;

    // init cuda data of long part
    uint32_t *dlong_val;
    MAT_VAL_TYPE *dval_by_warp;
    MAT_PTR_TYPE *dlong_ptr_warp;
    int *dlong_cid; 
    int *drid_by_warp;

    // init cuda data of short part
    uint32_t *dshort_val;
    int *dshort_cid;

    // init cuda data of reg & irreg part
    uint32_t *dreg_val;
    uint32_t *dirreg_val;
    MAT_PTR_TYPE *dblock_ptr, *dirreg_rpt;
    int *dreg_cid, *dirreg_cid;

    cudaMalloc((void **)&dX_val, sizeof(MAT_VAL_TYPE) * (((colA + 1) / 2) * 2));
    cudaMalloc((void **)&dY_val, sizeof(MAT_VAL_TYPE) * (((rowA + 1) / 2) * 2));
    cudaMemcpy(dX_val, X_val, sizeof(MAT_VAL_TYPE) * (((colA + 1) / 2) * 2), cudaMemcpyHostToDevice);
    cudaMemset(dY_val, 0.0, sizeof(MAT_VAL_TYPE) * (((rowA + 1) / 2) * 2));

    // cudaMalloc((void **)&dlong_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_long); 
    cudaMalloc((void **)&dlong_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_long); 
    cudaMalloc((void **)&dlong_cid, sizeof(int) * fill0_nnz_long);
    cudaMalloc((void **)&drid_by_warp, sizeof(int) * warp_number);
    cudaMalloc((void **)&dval_by_warp, sizeof(MAT_VAL_TYPE) * warp_number);
    cudaMalloc((void **)&dlong_ptr_warp, sizeof(MAT_PTR_TYPE) * (row_long + 1));
    cudaMemcpy(dlong_val, long_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_long, cudaMemcpyHostToDevice);
    cudaMemcpy(dlong_cid, long_cid, sizeof(int) * fill0_nnz_long, cudaMemcpyHostToDevice);
    cudaMemcpy(drid_by_warp, rid_by_warp, sizeof(int) * warp_number, cudaMemcpyHostToDevice);
    cudaMemcpy(dlong_ptr_warp, long_rpt_new, sizeof(MAT_PTR_TYPE) * (row_long + 1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dshort_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_short);
    cudaMalloc((void **)&dshort_cid, sizeof(int) * fill0_nnz_short);
    cudaMemcpy(dshort_val, short_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_short, cudaMemcpyHostToDevice);
    cudaMemcpy(dshort_cid, short_cid, sizeof(int) * fill0_nnz_short, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dreg_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_reg);
    cudaMalloc((void **)&dreg_cid, sizeof(int) * fill0_nnz_reg);
    cudaMalloc((void **)&dblock_ptr, sizeof(MAT_PTR_TYPE) * (blocknum + 1));
    cudaMemcpy(dreg_val, reg_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_reg, cudaMemcpyHostToDevice);
    cudaMemcpy(dreg_cid, reg_cid, sizeof(int) * fill0_nnz_reg, cudaMemcpyHostToDevice);
    cudaMemcpy(dblock_ptr, blockPtr, sizeof(MAT_PTR_TYPE) * (blocknum + 1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dirreg_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_irreg);
    cudaMalloc((void **)&dirreg_rpt, sizeof(MAT_PTR_TYPE) * (row_block + 1));
    cudaMalloc((void **)&dirreg_cid, sizeof(int) * nnz_irreg);
    cudaMemcpy(dirreg_val, irreg_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_irreg, cudaMemcpyHostToDevice);
    cudaMemcpy(dirreg_rpt, irreg_rpt, sizeof(MAT_PTR_TYPE) * (row_block + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dirreg_cid, irreg_cid, sizeof(int) * nnz_irreg, cudaMemcpyHostToDevice); 
    
    int carveout = 0;
    cudaFuncSetAttribute(dasp_spmv<1>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv<2>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv<4>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<1>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<2>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<4>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    
    int warmup_time = 100;
    int execute_time = 1000;
    if (rowloop == 1)
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv<1><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv<1><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<1><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t3, NULL);

    }
    else if (rowloop == 2)
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv<2><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv<2><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<2><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t3, NULL);
    }
    else
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34, fill0_nnz_short22);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t3, NULL);
    }

    double dasp_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time; 
    double dasp_gflops = (double)((long)nnzA * 2) / (dasp_time * 1e6);
    double dasp_time_bypass = ((t3.tv_sec - t2.tv_sec) * 1000.0 + (t3.tv_usec - t2.tv_usec) / 1000.0) / execute_time; 
    double dasp_gflops_bypass = (double)((long)nnzA * 2) / (dasp_time_bypass * 1e6);
    double dasp_bandwidth1 = (double)data_X / (dasp_time_bypass * 1e6);
    double dasp_bandwidth2 = (double)data_X2 / (dasp_time_bypass * 1e6);
    printf("SpMV_X:  %8.4lf ms, %8.4lf GFlop/s, %9.4lf GB/s, %9.4lf GB/s\n", dasp_time, dasp_gflops, dasp_bandwidth1, dasp_bandwidth2);
    printf("SpMV_X2: %8.4lf ms, %8.4lf GFlop/s, %9.4lf GB/s, %9.4lf GB/s\n", dasp_time_bypass, dasp_gflops_bypass, dasp_bandwidth1, dasp_bandwidth2);

    // printf("\nrowA = %d, row_long = %d, row_block = %d, row_short1 = %d, common13 = %d, row_short_3 = %d, row_short_4 = %d, row_short_2 = %d\n", rowA, row_long, row_block, short_row_1, common_13, short_row_3, short_row_4, short_row_2);

    cudaMemcpy(Y_val, dY_val, sizeof(MAT_VAL_TYPE) * rowA, cudaMemcpyDeviceToHost);

    cudaFree(dX_val);
    cudaFree(dY_val);

    cudaFree(dlong_val);
    cudaFree(dlong_cid);
    cudaFree(dval_by_warp);
    cudaFree(drid_by_warp);
    cudaFree(dlong_ptr_warp);

    cudaFree(dshort_cid);
    cudaFree(dshort_val);

    cudaFree(dreg_val);
    cudaFree(dreg_cid);
    cudaFree(dblock_ptr);
    cudaFree(dirreg_cid);
    cudaFree(dirreg_rpt);
    cudaFree(dirreg_val);

    /* write the device result y */
    // FILE* fp_y;
    // fp_y = fopen("y_cuda.txt", "w");
    // for (int i = 0; i < rowA; i++)
    // {
    //     float temp_val = Y_val[i];
    //     fprintf(fp_y, "%f \n", temp_val);
    // }
    // fclose(fp_y);

    // int result = verify(Y_val2, Y_val, rowA, 1);

    FILE* fout;
    fout = fopen("data/spmv_f16_record.csv", "a");
    fprintf(fout, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", filename, rowA, colA, nnzA, short_row_1, common_13, short_row_3, short_row_4, short_row_2, row_long, row_block, nnz_short, fill0_nnz_short, nnz_long, fill0_nnz_long, origin_nnz_reg, fill0_nnz_reg, nnz_irreg);
    fprintf(fout, "%lf,%d,%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf,", rate_fill0, block_longest, data_X, dasp_pre, dasp_time, dasp_gflops, dasp_time_bypass, dasp_gflops_bypass, dasp_bandwidth1, dasp_bandwidth2);
    fclose(fout);

    printf("\n");

    free(short_rid_1);
    free(short_rid_2);
    free(short_rid_3);
    free(short_rid_4);
    free(long_rid);
    free(zero_rid);
    free(ridA);

    free(rptA);
    free(long_rpt);

    free(short_val);
    free(short_cid);

    free(long_cid);
    free(long_val);
    free(long_rpt_new);
    free(val_by_warp);
    free(rid_by_warp);

    free(reg_val);
    free(reg_cid);
    free(blockPtr);

    free(irreg_rpt);
    free(irreg_cid);
    free(irreg_val);
}