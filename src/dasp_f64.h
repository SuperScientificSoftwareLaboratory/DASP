/*
    第一版整合版本，统一一个线程块4个warp
    long part： 一个warp进行两次mma，多个block得到一个y
    row_block part：
        根据row-block行数进行分情况讨论
        row-block < 59990                        一个warp算1个行块
        row-block >= 59990 && row-block < 400000 一个warp算2个行块
        row-block >= 400000                      一个warp算4个行块
        
    short part：一个warp进行四次mma，一个warp得到32个y

    增加 bypass L1 cache
*/

#include "common.h"
#include "utils.h"

#define groupNum 1
#define warpNum_short 4
#define loopNum_short 4
#define warpNum_long 4
#define loopNum_long 2


__device__ __forceinline__ MAT_VAL_TYPE warpReduceSum(MAT_VAL_TYPE sum){
    sum += __shfl_down_sync(0xffffffff,sum,16);
    sum += __shfl_down_sync(0xffffffff,sum,8);
    sum += __shfl_down_sync(0xffffffff,sum,4);
    sum += __shfl_down_sync(0xffffffff,sum,2);
    sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}

__device__ __forceinline__ MAT_VAL_TYPE load_double_from_global(const MAT_VAL_TYPE* a)
{
    MAT_VAL_TYPE r;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ void store_double_to_global(const MAT_VAL_TYPE* a, MAT_VAL_TYPE v)
{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(a), "d"(v));
}

__device__ __forceinline__ int load_int_from_global(const int* a)
{
    int r;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}

__global__ void longPart_sum(int *dlongA_rpt, MAT_VAL_TYPE *dwarp_val, MAT_VAL_TYPE *dY_val, int row_long)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int laneid = 31 & tid;
    int global_warpid = bid * warpNum_long + (tid >> 5);

    if (global_warpid >= row_long) return;

    int offset_longA = load_int_from_global(dlongA_rpt + global_warpid);
    MAT_VAL_TYPE *cur_temp_val = dwarp_val + offset_longA;
    int len = load_int_from_global(dlongA_rpt + global_warpid + 1) - offset_longA;

    MAT_VAL_TYPE thread_val = 0;
    for (int i = laneid; i < len; i += WARP_SIZE)
    {
        thread_val += load_double_from_global(cur_temp_val + i);
    }
    thread_val = warpReduceSum(thread_val);

    if (laneid == 0)
        store_double_to_global(dY_val + global_warpid, thread_val);
}

template <int rowloop>  // this parameter must be 1 or 2 or 4
__global__ void dasp_spmv2(MAT_VAL_TYPE *dX_val, MAT_VAL_TYPE *dY_val,
                          MAT_VAL_TYPE *dlongA_val, int *dlongA_cid, MAT_VAL_TYPE *dwarp_val, MAT_PTR_TYPE *dlongA_rpt, int row_long,
                          MAT_VAL_TYPE *dregA_val, int *dregA_cid, MAT_PTR_TYPE *dblockA_ptr, int row_block, int blocknum, 
                          MAT_VAL_TYPE *dirregA_val, int *dirregA_cid, MAT_PTR_TYPE *dirregA_rpt,
                          MAT_VAL_TYPE *dshort_val, int *dshort_cid, int short_row_1, int common_13, int short_row_34, int short_row_2,
                          int offset_reg, int offset_short1, int offset_short13, int offset_short34, int offset_short22,
                          MAT_PTR_TYPE fill0_nnz_short13, MAT_PTR_TYPE fill0_nnz_short34)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int laneid = 31 & tid;
 
    if (bid < offset_reg)
    {
        // long part
        int global_warpid = bid * warpNum_long + (tid >> 5);
        int offset = global_warpid * loopNum_long * MMA_M * MMA_K;
        MAT_VAL_TYPE *curA_val = dlongA_val + offset;
        int *curA_cid = dlongA_cid + offset;

        int groupID = laneid >> 2;
        int tID_in_group = 3 & laneid;

        MAT_VAL_TYPE fragA, fragB;
        MAT_VAL_TYPE fragC[2];
        fragC[0] = 0.0, fragC[1] = 0.0;

        int idx = tID_in_group + groupID * MMA_K;
        
        #pragma unroll
        for (int i = 0; i < loopNum_long; i++)
        {
            fragA = load_double_from_global(curA_val + idx);
            int x_idx = load_int_from_global(curA_cid + idx);
            fragB = dX_val[x_idx];

            mma_m8n8k4(fragC, fragA, fragB);
            idx += 32;
        }

        fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 9, 32);
        fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
        fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 9, 32);
        fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);
        fragC[0] += __shfl_sync(0xffffffff, fragC[1], 4);

        if (laneid == 0) 
            store_double_to_global(dwarp_val + global_warpid, fragC[0]);

        // if (global_warpid >= row_long) return;

        // // MAT_VAL_TYPE *cur_temp_val = dwarp_val + dlongA_rpt[global_warpid];
        // int offset_longA = load_int_from_global(dlongA_rpt + global_warpid);
        // MAT_VAL_TYPE *cur_temp_val = dwarp_val + offset_longA;
        // int len = load_int_from_global(dlongA_rpt + global_warpid + 1) - offset_longA;

        // MAT_VAL_TYPE thread_val = 0;
        // for (int i = laneid; i < len; i += WARP_SIZE)
        // {
        //     thread_val += load_double_from_global(cur_temp_val + i);
        // }
        // thread_val = warpReduceSum(thread_val);

        // if (laneid == 0)
        //     store_double_to_global(dY_val + global_warpid, thread_val);

    }
    else if (bid >= offset_reg && bid < offset_short1)
    {
        // row-block part
        int bid_reg = bid - offset_reg;
        int warp_local = tid >> 5;

        int groupID = laneid >> 2;
        int tID_in_group = 3 & laneid;
        MAT_VAL_TYPE fragA, fragB, fragC[2];

        if (rowloop == 1)
        {
            int block_idx = bid_reg * 4 + warp_local;
            // int offset_A = dblockA_ptr[block_idx];
            int offset_A = load_int_from_global(dblockA_ptr + block_idx);
            int blocklen = (load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A) >> 3;

            if (block_idx >= blocknum) return;

            MAT_VAL_TYPE *curA_val = dregA_val + offset_A;
            int *curA_cid = dregA_cid + offset_A;

            fragC[0] = 0.0, fragC[1] = 0.0;
            int idx = tID_in_group + groupID * MMA_K;
            for (int i = 0; i < blocklen; i += MMA_K)
            {
                fragA = load_double_from_global(curA_val + idx);
                int x_idx = load_int_from_global(curA_cid + idx);
                fragB = dX_val[x_idx];
                mma_m8n8k4(fragC, fragA, fragB);
                idx += 32;
            }

            int offset_y = block_idx * BlockSize + groupID;
            if (tID_in_group == (groupID >> 1) && offset_y < row_block)
            {
                store_double_to_global(dY_val + row_long + offset_y, fragC[1 & groupID]);
            }

            int cur_row = block_idx * BlockSize + laneid;
            if (laneid < BlockSize && cur_row < row_block)
            {
                MAT_VAL_TYPE cur_y = 0.0;
                // for (int i = dirregA_rpt[cur_row]; i < dirregA_rpt[cur_row + 1]; i ++)
                for (int i = dirregA_rpt[cur_row]; i < dirregA_rpt[cur_row + 1]; i ++)
                {
                    cur_y += load_double_from_global(dirregA_val + i) * dX_val[dirregA_cid[i]];
                }
                cur_y += load_double_from_global(dY_val + row_long + cur_row);
                store_double_to_global(dY_val + row_long + cur_row, cur_y);
            }
        }

        if (rowloop == 2)
        {
            MAT_VAL_TYPE res;
            #pragma unroll
            for (int i = 0; i < 2; i ++)
            {
                int block_idx = bid_reg * 8 + warp_local * 2 + i;
                int offset_A = load_int_from_global(dblockA_ptr + block_idx);
                int blocklen = (load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A) >> 3;
                // int offset_A = dblockA_ptr[block_idx];
                // int blocklen = (dblockA_ptr[block_idx + 1] - offset_A) >> 3;

                MAT_VAL_TYPE *curA_val = dregA_val + offset_A;
                int *curA_cid = dregA_cid + offset_A;

                fragC[0] = 0.0, fragC[1] = 0.0;
                int idx = tID_in_group + groupID * MMA_K;
                for (int j = 0; j < blocklen; j += MMA_K)
                {
                    fragA = load_double_from_global(curA_val + idx);
                    int x_idx = load_int_from_global(curA_cid + idx);
                    fragB = dX_val[x_idx];
                    mma_m8n8k4(fragC, fragA, fragB);
                    idx += 32;
                }
                int target_id = ((laneid - i * 8) >> 1) * 9;
                fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
                fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
                if ((laneid >> 3) == i) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            }

            int cur_row = bid_reg * 8 * BlockSize + warp_local * 2 * BlockSize + laneid;
            if (laneid < 16 && cur_row < row_block)
            {
                for (int i = dirregA_rpt[cur_row]; i < dirregA_rpt[cur_row + 1]; i ++)
                {
                    res += load_double_from_global(dirregA_val + i) * dX_val[dirregA_cid[i]];
                }
                store_double_to_global(dY_val + row_long + cur_row, res);
            }
        }

        if (rowloop == 4)
        {
            MAT_VAL_TYPE res;
            #pragma unroll
            for (int i = 0; i < 4; i ++)
            {
                int block_idx = bid_reg * 16 + warp_local * 4 + i;
                // int offset_A = dblockA_ptr[block_idx];
                // int blocklen = (dblockA_ptr[block_idx + 1] - offset_A) >> 3;
                int offset_A = load_int_from_global(dblockA_ptr + block_idx);
                int blocklen = (load_int_from_global(dblockA_ptr + block_idx + 1) - offset_A) >> 3;

                MAT_VAL_TYPE *curA_val = dregA_val + offset_A;
                int *curA_cid = dregA_cid + offset_A;

                fragC[0] = 0.0, fragC[1] = 0.0;
                int idx = tID_in_group + groupID * MMA_K;
                for (int j = 0; j < blocklen; j += MMA_K)
                {
                    fragA = load_double_from_global(curA_val + idx);
                    int x_idx = load_int_from_global(curA_cid + idx);
                    fragB = dX_val[x_idx];
                    mma_m8n8k4(fragC, fragA, fragB);
                    idx += 32;
                }
                int target_id = ((laneid - i * 8) >> 1) * 9;
                fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
                fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
                if ((laneid >> 3) == i) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            }
            int cur_row = bid_reg * 16 * BlockSize + warp_local * 4 * BlockSize + laneid;
            if (cur_row < row_block)
            {
                for (int i = dirregA_rpt[cur_row]; i < dirregA_rpt[cur_row + 1]; i ++)
                {
                    res += load_double_from_global(dirregA_val + i) * dX_val[dirregA_cid[i]];
                }
                store_double_to_global(dY_val + row_long + cur_row, res);
            }
        }
    }
    else if (bid >= offset_short1 && bid < offset_short13)
    {
        // short part - 1 nnz/row
        int bid1 = bid - offset_short1;
        int tid1 = bid1 * blockDim.x + tid;
        if (tid1 >= short_row_1)
        {
            return;
        }
        
        int x_idx = load_int_from_global(dshort_cid + tid1);
        MAT_VAL_TYPE temp_y = load_double_from_global(dshort_val + tid1) * dX_val[x_idx];
        store_double_to_global(dY_val + row_long + row_block + tid1, temp_y);

    }
    else if (bid >= offset_short13 && bid < offset_short34)
    {
        // short part - block 1&3
        int warpid_local = tid >> 5;
        int bid13 = bid - offset_short13;

        MAT_VAL_TYPE fragA = 0.0, fragB = 0.0, fragC[2], res;
        int groupID = laneid >> 2;
        int tID_in_group = 3 & laneid;

        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            int offset = short_row_1 + ((bid13 * groupNum + i) * warpNum_short + warpid_local) * MMA_M * MMA_K * 2;
            MAT_VAL_TYPE *cur_val = dshort_val + offset;
            int *cur_cid = dshort_cid + offset;
            int idx = tID_in_group + groupID * MMA_K;
            
            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            int cid = load_int_from_global(cur_cid + idx);
            fragB = tID_in_group == 0 ? dX_val[cid] : 0;
            mma_m8n8k4(fragC, fragA, fragB);
            int target_id = (laneid >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid < 8) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragB = tID_in_group == 0 ? 0 : dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 8) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >> 3 == 1) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            idx += 32;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            cid = load_int_from_global(cur_cid + idx);
            fragB = tID_in_group == 0 ? dX_val[cid] : 0;
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 16) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >> 3 == 2) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragB = tID_in_group == 0 ? 0 : dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 24) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >= 24) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];

            int offset_y = ((bid13 * groupNum + i) * warpNum_short  + warpid_local) * WARP_SIZE + laneid;
            if (offset_y < common_13 * 2) 
                store_double_to_global(dY_val + row_long + row_block + short_row_1 + offset_y, res);

        }
    }
    else if (bid >= offset_short34 && bid < offset_short22)
    {
        // short part - block3 & block4
        int warpid_local = tid >> 5;
        int bid34 = bid - offset_short34;

        MAT_VAL_TYPE fragA = 0.0, fragB = 0.0, fragC[2], res;
        int groupID = laneid >> 2;
        int tID_in_group = 3 & laneid;

        #pragma unroll
        for (int j = 0; j < groupNum; j ++)
        {
            int offset = short_row_1 + fill0_nnz_short13 + ((bid34 * groupNum + j) * warpNum_short + warpid_local) * MMA_M * MMA_K * loopNum_short;
            MAT_VAL_TYPE *cur_val = dshort_val + offset;
            int *cur_cid = dshort_cid + offset;
            int idx = tID_in_group + groupID * MMA_K;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            int cid = load_int_from_global(cur_cid + idx);
            fragB = dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            int target_id = (laneid >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid < 8) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            idx += 32;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            cid = load_int_from_global(cur_cid + idx);
            fragB = dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 8) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >> 3 == 1) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            idx += 32;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            cid = load_int_from_global(cur_cid + idx);
            fragB = dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 16) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >> 3 == 2) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            idx += 32;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            cid = load_int_from_global(cur_cid + idx);
            fragB = dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 24) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >= 24) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            
            int offset_y = ((bid34 * groupNum + j) * warpNum_short + warpid_local) * WARP_SIZE + laneid;
            if (offset_y < short_row_34) 
                store_double_to_global(dY_val + row_long + row_block + short_row_1 + common_13 * 2 + offset_y, res);

        }
    }
    else
    {
        // short part - blocl 2&2
        int warpid_local = tid >> 5;
        int bid22 = bid - offset_short22;

        MAT_VAL_TYPE fragA = 0.0, fragB = 0.0, fragC[2], res;
        int groupID = laneid >> 2;
        int tID_in_group = 3 & laneid;
        
        #pragma unroll
        for (int i = 0; i < groupNum; i ++)
        {
            int offset = short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + ((bid22 * groupNum + i) * warpNum_short + warpid_local) * MMA_M * MMA_K * 2;
            MAT_VAL_TYPE *cur_val = dshort_val + offset;
            int *cur_cid = dshort_cid + offset;
            int idx = tID_in_group + groupID * MMA_K;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            int cid = load_int_from_global(cur_cid + idx);
            fragB = tID_in_group < 2 ? dX_val[cid] : 0;
            mma_m8n8k4(fragC, fragA, fragB);
            int target_id = (laneid >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid < 8) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragB = tID_in_group < 2 ? 0 : dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 8) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >> 3 == 1) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];
            idx += 32;

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragA = load_double_from_global(cur_val + idx);
            cid = load_int_from_global(cur_cid + idx);
            fragB = tID_in_group < 2 ? dX_val[cid] : 0;
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 16) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >> 3 == 2) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];

            fragC[0] = 0.0, fragC[1] = 0.0;
            fragB = tID_in_group < 2 ? 0 : dX_val[cid];
            mma_m8n8k4(fragC, fragA, fragB);
            target_id = ((laneid - 24) >> 1) * 9;
            fragC[0] = __shfl_sync(0xffffffff, fragC[0], target_id);
            fragC[1] = __shfl_sync(0xffffffff, fragC[1], target_id + 4);
            if (laneid >= 24) res = (1 & laneid) == 0 ? fragC[0] : fragC[1];

            int offset_y = ((bid22 * groupNum + i) * warpNum_short + warpid_local) * WARP_SIZE + laneid;
            if (offset_y < short_row_2) 
                store_double_to_global(dY_val + row_long + row_block + short_row_1 + common_13 * 2 + short_row_34 + offset_y, res);
        }
    }
}

__host__ void spmv_all(char *filename, MAT_VAL_TYPE *csrValA, MAT_PTR_TYPE *csrRowPtrA, int *csrColIdxA, 
                      MAT_VAL_TYPE *X_val, MAT_VAL_TYPE *Y_val, int *order_rid, int rowA, int colA, MAT_PTR_TYPE nnzA, int NUM, double threshold, int block_longest)
{
    struct timeval t1, t2;

    // three parts: short row (1 & 3 & 2 & 4), long row, row-block (regular（origin & fill0） & irregular)
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
        common_13 = BlockSize * (common_13 / BlockSize);
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

    // int warpnum13 = (short_block13 + groupNum * 2 - 1) / (groupNum * 2); 
    // int warpnum34 = (short_block34 + groupNum * loopNum_short - 1) / (groupNum * loopNum_short);
    // int warpnum22 = (short_block22 + groupNum * 2 - 1) / (groupNum * 2); 

    int block13_per_threadblock = warpNum_short * groupNum * 2;
    int block22_per_threadblock = warpNum_short * groupNum * 2;
    int block34_per_threadblock = warpNum_short * groupNum * loopNum_short;

    int threadblock13 = (short_block13 + block13_per_threadblock - 1) / block13_per_threadblock;
    int threadblock22 = (short_block22 + block22_per_threadblock - 1) / block22_per_threadblock;
    int threadblock34 = (short_block34 + block34_per_threadblock - 1) / block34_per_threadblock;

    MAT_PTR_TYPE fill0_nnz_short13 = threadblock13 * block13_per_threadblock * MMA_M * MMA_K;
    MAT_PTR_TYPE fill0_nnz_short34 = threadblock34 * block34_per_threadblock * MMA_M * MMA_K;
    MAT_PTR_TYPE fill0_nnz_short22 = threadblock22 * block22_per_threadblock * MMA_M * MMA_K;
    MAT_PTR_TYPE fill0_nnz_short = short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + fill0_nnz_short22;
    MAT_VAL_TYPE *short_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * fill0_nnz_short);
    int *short_cid = (int *)malloc(sizeof(int) * fill0_nnz_short);
    memset(short_val, 0.0, sizeof(MAT_VAL_TYPE) * fill0_nnz_short);
    memset(short_cid, 0, sizeof(int) * fill0_nnz_short);

    int super_group = 1 + threadblock13 + threadblock34 + threadblock22;
    MAT_PTR_TYPE *superX_ptr = (MAT_PTR_TYPE *)malloc(sizeof(int) * (super_group + 1));
    
    for (int i = 0; i < short_row_1; i ++)
    {
        int cur_row = short_rid_1[i];
        short_val[i] = csrValA[csrRowPtrA[cur_row]];
        short_cid[i] = csrColIdxA[csrRowPtrA[cur_row]];
    }

    for (int i = 0; i < short_block13; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + short_row_1 + i * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + short_row_1 + i * MMA_M * MMA_K;

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

    for (int i = 0; i < short_row_3; i ++)
    {
        // MAT_VAL_TYPE *cur_short_val = short_val + short_row_1 + short_block13 * MMA_M * MMA_K + i * MMA_K;
        // int *cur_short_cid = short_cid + short_row_1 + short_block13 * MMA_M * MMA_K + i * MMA_K;
        MAT_VAL_TYPE *cur_short_val = short_val + short_row_1 + fill0_nnz_short13 + i * MMA_K;
        int *cur_short_cid = short_cid + short_row_1 + fill0_nnz_short13 + i * MMA_K;
        
        int cur_row = short_rid_3[common_13 + i];

        cur_short_val[0] = csrValA[csrRowPtrA[cur_row]];
        cur_short_val[1] = csrValA[csrRowPtrA[cur_row] + 1]; 
        cur_short_val[2] = csrValA[csrRowPtrA[cur_row] + 2]; 
        cur_short_cid[0] = csrColIdxA[csrRowPtrA[cur_row]];
        cur_short_cid[1] = csrColIdxA[csrRowPtrA[cur_row] + 1]; 
        cur_short_cid[2] = csrColIdxA[csrRowPtrA[cur_row] + 2]; 
    }

    for (int i = 0; i < short_row_4; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + short_row_1 + fill0_nnz_short13 + (short_row_3 + i) * MMA_K;
        int *cur_short_cid = short_cid + short_row_1 + fill0_nnz_short13 + (short_row_3 + i) * MMA_K;
        
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

    for (int i = 0; i < short_block22; i ++)
    {
        MAT_VAL_TYPE *cur_short_val = short_val + short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + i * MMA_M * MMA_K;
        int *cur_short_cid = short_cid + short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + i * MMA_M * MMA_K;

        for (int j = 0; j < BlockSize * 2 && (i * BlockSize * 2 + j) < short_row_2; j ++)
        {
            int cur_row = short_rid_2[i * BlockSize * 2 + j];
            cur_short_val[j % BlockSize * MMA_K + (j / BlockSize) * 2] = csrValA[csrRowPtrA[cur_row]];
            cur_short_val[j % BlockSize * MMA_K + (j / BlockSize) * 2 + 1] = csrValA[csrRowPtrA[cur_row] + 1];
            cur_short_cid[j % BlockSize * MMA_K + (j / BlockSize) * 2] = csrColIdxA[csrRowPtrA[cur_row]];
            cur_short_cid[j % BlockSize * MMA_K + (j / BlockSize) * 2 + 1] = csrColIdxA[csrRowPtrA[cur_row] + 1];
        }
    }

    int *short_cid_temp = (int *)malloc(sizeof(int) * fill0_nnz_short);
    memcpy(short_cid_temp, short_cid, sizeof(int) * fill0_nnz_short);

    quick_sort_key(short_cid_temp, short_row_1);
    int nnzr = short_row_1 > 0 ? 1 : 0;
    for (int i = 1; i < short_row_1; i ++)
    {
        nnzr += short_cid_temp[i] != short_cid_temp[i - 1] ? 1 : 0;
    }
    superX_ptr[0] = nnzr;

    MAT_PTR_TYPE *cur_superX_ptr = superX_ptr + 1;
    for (int i = 0; i < threadblock13; i++)
    {
        int *cur_short_cid_temp = short_cid_temp + short_row_1 + i * block13_per_threadblock * MMA_M * MMA_K;
        int len = block13_per_threadblock * MMA_M * MMA_K;
        quick_sort_key(cur_short_cid_temp, len);
        int nnzcid = len > 0 ? 1 : 0;
        for (int j = 1; j < len; j ++)
        {
            nnzcid += cur_short_cid_temp[j] != cur_short_cid_temp[j - 1] ? 1 : 0;
        }
        cur_superX_ptr[i] = nnzcid;
    }

    cur_superX_ptr = superX_ptr + 1 + threadblock13;
    for (int i = 0; i < threadblock34; i++)
    {
        int *cur_short_cid_temp = short_cid_temp + short_row_1 + fill0_nnz_short13 + i * block34_per_threadblock * MMA_M * MMA_K;
        int len = block34_per_threadblock * MMA_M * MMA_K;
        quick_sort_key(cur_short_cid_temp, len);
        int nnzcid = len > 0 ? 1 : 0;
        for (int j = 1; j < len; j ++)
        {
            nnzcid += cur_short_cid_temp[j] != cur_short_cid_temp[j - 1] ? 1 : 0;
        }
        cur_superX_ptr[i] = nnzcid;
    }

    cur_superX_ptr = superX_ptr + 1 + threadblock13 + threadblock34;
    for (int i = 0; i < threadblock22; i++)
    {
        int *cur_short_cid_temp = short_cid_temp + short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + i * block22_per_threadblock * MMA_M * MMA_K;
        int len = block22_per_threadblock * MMA_M * MMA_K;
        quick_sort_key(cur_short_cid_temp, len);
        int nnzcid = len > 0 ? 1 : 0;
        for (int j = 1; j < len; j ++)
        {
            nnzcid += cur_short_cid_temp[j] != cur_short_cid_temp[j - 1] ? 1 : 0;
        }
        cur_superX_ptr[i] = nnzcid;
    }
    exclusive_scan(superX_ptr, super_group + 1);
    MAT_PTR_TYPE nnz_superX = superX_ptr[super_group];

    int new_cid_len = short_row_1 + threadblock13 * block13_per_threadblock * MMA_M * MMA_K / 4 + \
                                    threadblock34 * block34_per_threadblock * MMA_M * MMA_K / 4 + \
                                    threadblock22 * block22_per_threadblock * MMA_M * MMA_K / 4;

    int *short_cid_new = (int *)malloc(sizeof(int) * new_cid_len);

    int *superX_cid = (int *)malloc(sizeof(int) * nnz_superX);
    int flag = 0;
    if (short_row_1)
    {
        superX_cid[0] = short_cid_temp[0];
        flag ++;
    }
    for (int j = 1; j < short_row_1; j ++)
    {
        if (short_cid_temp[j] != short_cid_temp[j - 1])
        {
            superX_cid[flag] = short_cid_temp[j];
            flag ++;
        }
    }
    if (flag != superX_ptr[1]) printf("flag1 = %d, len = %d\n", flag, superX_ptr[1]);
    for (int i = 0; i < short_row_1; i ++)
    {
        short_cid_new[i] = BinarySearch(superX_cid, superX_ptr[1], short_cid[i]);
    }

    cur_superX_ptr = superX_ptr + 1;
    for (int i = 0; i < threadblock13; i ++)
    {
        int *cur_short_cid_temp = short_cid_temp + short_row_1 + i * block13_per_threadblock * MMA_M * MMA_K;
        int len = block13_per_threadblock * MMA_M * MMA_K;
        int *cur_superX_cid = superX_cid + cur_superX_ptr[i];
        int xlen = cur_superX_ptr[i + 1] - cur_superX_ptr[i];
        int flag_cid = 0;
        if (len)
        {
            cur_superX_cid[0] = cur_short_cid_temp[0];
            flag_cid ++;
        }
        else
        {
            continue;
        }
        for (int j = 1; j < len; j ++)
        {
            if (cur_short_cid_temp[j] != cur_short_cid_temp[j - 1])
            {
                cur_superX_cid[flag_cid] = cur_short_cid_temp[j];
                flag_cid ++;
            }
        }
        if (flag_cid != xlen) printf("flag13 = %d, len = %d\n", flag_cid, xlen);

        int *cur_short_cid_new = short_cid_new + short_row_1 + i * (block13_per_threadblock * MMA_M * MMA_K / 4);
        int *cur_short_cid = short_cid + short_row_1 + i * block13_per_threadblock * MMA_M * MMA_K;
        for (int j = 0; j < len; j ++)
        {
            // cur_short_cid_new[j] = BinarySearch(cur_superX_cid, xlen, cur_short_cid[j]);
            SET_8_BIT(cur_short_cid_new[j / 4], BinarySearch(cur_superX_cid, xlen, cur_short_cid[j]), j % 4);
        }
    }

    cur_superX_ptr = superX_ptr + 1 + threadblock13;
    for (int i = 0; i < threadblock34; i ++)
    {
        int *cur_short_cid_temp = short_cid_temp + short_row_1 + fill0_nnz_short13 + i * block34_per_threadblock * MMA_M * MMA_K;
        int len = block34_per_threadblock * MMA_M * MMA_K;
        int *cur_superX_cid = superX_cid + cur_superX_ptr[i];
        int xlen = cur_superX_ptr[i + 1] - cur_superX_ptr[i];
        int flag_cid = 0;
        if (len)
        {
            cur_superX_cid[0] = cur_short_cid_temp[0];
            flag_cid ++;
        }
        else
        {
            continue;
        }
        for (int j = 1; j < len; j ++)
        {
            if (cur_short_cid_temp[j] != cur_short_cid_temp[j - 1])
            {
                cur_superX_cid[flag_cid] = cur_short_cid_temp[j];
                flag_cid ++;
            }
        }
        if (flag_cid != xlen) printf("flag34 = %d, len = %d\n", flag_cid, xlen);

        int *cur_short_cid_new = short_cid_new + short_row_1 + fill0_nnz_short13 / 4 + i * (block34_per_threadblock * MMA_M * MMA_K / 4);
        int *cur_short_cid = short_cid + short_row_1 + fill0_nnz_short13 + i * block34_per_threadblock * MMA_M * MMA_K;
        for (int j = 0; j < len; j ++)
        {
            // cur_short_cid_new[j] = BinarySearch(cur_superX_cid, xlen, cur_short_cid[j]);
            SET_8_BIT(cur_short_cid_new[j / 4], BinarySearch(cur_superX_cid, xlen, cur_short_cid[j]), j % 4);
        }
    }

    cur_superX_ptr = superX_ptr + 1 + threadblock13 + threadblock34;
    for (int i = 0; i < threadblock22; i ++)
    {
        int *cur_short_cid_temp = short_cid_temp + short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + i * block22_per_threadblock * MMA_M * MMA_K;
        int len = block22_per_threadblock * MMA_M * MMA_K;
        int *cur_superX_cid = superX_cid + cur_superX_ptr[i];
        int xlen = cur_superX_ptr[i + 1] - cur_superX_ptr[i];
        int flag_cid = 0;
        if (len)
        {
            cur_superX_cid[0] = cur_short_cid_temp[0];
            flag_cid ++;
        }
        else
        {
            continue;
        }
        for (int j = 1; j < len; j ++)
        {
            if (cur_short_cid_temp[j] != cur_short_cid_temp[j - 1])
            {
                cur_superX_cid[flag_cid] = cur_short_cid_temp[j];
                flag_cid ++;
            }
        }
        if (flag_cid != xlen) printf("flag22 = %d, len = %d\n", flag_cid, xlen);

        int *cur_short_cid_new = short_cid_new + short_row_1 + (fill0_nnz_short13 + fill0_nnz_short34) / 4 + i * (block22_per_threadblock * MMA_M * MMA_K / 4);
        int *cur_short_cid = short_cid + short_row_1 + fill0_nnz_short13 + fill0_nnz_short34 + i * block22_per_threadblock * MMA_M * MMA_K;
        for (int j = 0; j < len; j ++)
        {
            // cur_short_cid_new[j] = BinarySearch(cur_superX_cid, xlen, cur_short_cid[j]);
            SET_8_BIT(cur_short_cid_new[j / 4], BinarySearch(cur_superX_cid, xlen, cur_short_cid[j]), j % 4);
        }
    }

    MAT_VAL_TYPE *superX_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * nnz_superX);
    for (int i = 0; i < nnz_superX; i ++)
    {
        superX_val[i] = X_val[superX_cid[i]];
    }

    // printf("superX_nnz = %d, cid_len = %d\n", nnz_superX, new_cid_len);

    // resort except rows
    radix_sort(rptA, ridA, row_block);
   
    // int sort_group_len = block_longest / 4;
    // int *sort_rpt = (int *)malloc(sizeof(int) * (sort_group_len + 1));
    // memset(sort_rpt, 0, sizeof(int) * (sort_group_len + 1));
    // for (int i = 0; i < row_block; i ++)
    // {
    //     sort_rpt[rptA[i] / 4] ++;
    // }
    // exclusive_scan(sort_rpt, sort_group_len + 1);
    
    // #pragma omp parallel for
    // for (int i = 0; i < sort_group_len; i ++)
    // {
    //     int group_len = sort_rpt[i + 1] - sort_rpt[i];
    //     quick_sort_key_idx(ridA + sort_rpt[i], rptA + sort_rpt[i], group_len);
    // }

    // FILE* fp_rid;
    // fp_rid = fopen("rid.txt", "w");
    // for (int i = 0; i < sort_group_len; ++ i)
    // {
    //     int group_len = sort_rpt[i + 1] - sort_rpt[i];
    //     fprintf(fp_rid, "i = %d, len = %d: ", i, group_len);
    //     for (int j = 0; j < group_len; j ++)
    //     {
    //         fprintf(fp_rid, "%d ", ridA[sort_rpt[i] + j]);
    //     }
    //     fprintf(fp_rid, "\n");
    // }
    // fclose(fp_rid);

    // free(sort_rpt);

    // printf("resort over!\n");

    

    // get the data except short row part
    // (rptA, cidA, valA)
    exclusive_scan(rptA, row_block + 1);
    exclusive_scan(long_rpt, row_long + 1);
    // nnz_row_block = rptA[row_block];
    nnz_long = long_rpt[row_long];

    //record the sort order
    memcpy(order_rid, long_rid, sizeof(int) * row_long);
    memcpy(order_rid + row_long, ridA, sizeof(int) * row_block);
    memcpy(order_rid + row_long + row_block, short_rid_1, sizeof(int) * short_row_1);
    for (int i = 0; i < short_block13; i ++)
    {
        int *cur_order_rid = order_rid + row_long + row_block + short_row_1 + i * BlockSize * 2;

        for (int j = 0; j < BlockSize; j ++)
        {
            cur_order_rid[j] = short_rid_1[short_row_1 + i * BlockSize + j];
            cur_order_rid[BlockSize + j] = short_rid_3[i * BlockSize + j];
        } 
    }
    memcpy(order_rid + row_long + row_block + short_row_1 + common_13 * 2, short_rid_3 + common_13, sizeof(int) * short_row_3);
    memcpy(order_rid + row_long + row_block + short_row_1 + common_13 * 2 + short_row_3, short_rid_4, sizeof(int) * short_row_4);
    memcpy(order_rid + row_long + row_block + short_row_1 + common_13 * 2 + short_row_3 + short_row_4, short_rid_2, sizeof(int) * short_row_2);
    memcpy(order_rid + row_long + row_block + short_row_1 + common_13 * 2 + short_row_3 + short_row_4 + short_row_2, zero_rid, sizeof(int) * row_zero);

    int short_row = short_row_1 + common_13 * 2 + short_row_34 + short_row_2;
    int offset_short_row = row_long + row_block;

    MAT_VAL_TYPE *short3_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * nnz_short);
    int *short3_cid = (int *)malloc(sizeof(int) * nnz_short);
    MAT_PTR_TYPE *short3_rpt = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (short_row + 1));

    for (int i = 0; i < short_row; i ++)
    {
        int idx = order_rid[offset_short_row + i];
        short3_rpt[i] = csrRowPtrA[idx + 1] - csrRowPtrA[idx];
    }
    exclusive_scan(short3_rpt, short_row + 1);

    for (int i = 0; i < short_row; i ++)
    {
        int idx = order_rid[offset_short_row + i];
        memcpy(short3_val + short3_rpt[i], csrValA + csrRowPtrA[idx], sizeof(MAT_VAL_TYPE) * (csrRowPtrA[idx + 1] - csrRowPtrA[idx]));
        memcpy(short3_cid + short3_rpt[i], csrColIdxA + csrRowPtrA[idx], sizeof(int) * (csrRowPtrA[idx + 1] - csrRowPtrA[idx]));
    }

    // get the long part data
    MAT_PTR_TYPE *long_rpt_new = (MAT_PTR_TYPE *)malloc(sizeof(MAT_PTR_TYPE) * (row_long + 1));
    memset(long_rpt_new, 0, sizeof(MAT_PTR_TYPE) * (row_long + 1));
    int warp_number = 0;
    for (int i = 0; i < row_long; i ++)
    {
        int nnz_num = long_rpt[i + 1] - long_rpt[i];
        int cur_warp_num = (nnz_num + MMA_M * MMA_K * loopNum_long - 1) / (MMA_M * MMA_K * loopNum_long);
        // warp_number += cur_warp_num;
        long_rpt_new[i] = cur_warp_num;
    }
    exclusive_scan(long_rpt_new, row_long + 1);
    warp_number = long_rpt_new[row_long];

    int BlockNum_long = (warp_number + warpNum_long - 1) / warpNum_long;
    int fill0_nnz_long = BlockNum_long * warpNum_long * loopNum_long * MMA_M * MMA_K;
    warp_number = BlockNum_long * warpNum_long;
    MAT_VAL_TYPE *val_by_warp = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * warp_number);
    int *rid_by_warp = (int *)malloc(sizeof(int) * warp_number);
    MAT_VAL_TYPE *long_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * fill0_nnz_long);
    memset(long_val, 0.0, sizeof(MAT_VAL_TYPE) * fill0_nnz_long);
    int *long_cid = (int *)malloc(sizeof(int) * fill0_nnz_long);
    memset(long_cid, 0, sizeof(int) * fill0_nnz_long);

    // int count_warp = 0;
    for (int i = 0; i < row_long; i ++)
    {
        MAT_VAL_TYPE *cur_val = long_val + long_rpt_new[i] * loopNum_long * MMA_M * MMA_K;
        int *cur_cid = long_cid + long_rpt_new[i] * loopNum_long * MMA_M * MMA_K;
        int real_rid = long_rid[i];
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
    }
    
    exclusive_scan(blockPtr, blocknum + 1);
    exclusive_scan(irreg_rpt, row_block + 1);
    
    // int offset_row_block = row_long;
    fill0_nnz_reg = blockPtr[blocknum];
    nnz_irreg = irreg_rpt[row_block];
    origin_nnz_reg = nnzA - nnz_irreg - nnz_long - nnz_short;

    // get the row-block part data---irregular part
    MAT_VAL_TYPE *irreg_val = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * nnz_irreg);
    int *irreg_cid = (int *)malloc(sizeof(int) * nnz_irreg);
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
                int row_len = csrRowPtrA[real_id + 1] - A_start;
                for (int i = 0; i < blocklen; i ++)
                {
                    reg_val[regA_start + i] = i < row_len ? csrValA[A_start + i] : 0.0;
                    reg_cid[regA_start + i] = i < row_len ? csrColIdxA[A_start + i] : 0;
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

    long fill0_nnz = fill0_nnz_short + fill0_nnz_long + nnz_irreg + fill0_nnz_reg;
    double rate_fill0 = (double)(fill0_nnz - nnzA) / nnzA;
    
    long long int data_X = (rowA + colA) * sizeof(MAT_VAL_TYPE) + \
                           fill0_nnz_long * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + warp_number * sizeof(MAT_VAL_TYPE) + (row_long + 1) * sizeof(int) + \
                           fill0_nnz_short * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + \
                           fill0_nnz_reg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (blocknum + 1) * sizeof(MAT_PTR_TYPE) + \
                           nnz_irreg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (row_block + 1) * sizeof(MAT_PTR_TYPE);
    
    long long int data_X2 = (rowA + nnzA) * sizeof(MAT_VAL_TYPE) + \
                            fill0_nnz_long * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + warp_number * sizeof(MAT_VAL_TYPE) + (row_long + 1) * sizeof(int) + \
                            fill0_nnz_short * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + \
                            fill0_nnz_reg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (blocknum + 1) * sizeof(MAT_PTR_TYPE) + \
                            nnz_irreg * (sizeof(MAT_VAL_TYPE) + sizeof(int)) + (row_block + 1) * sizeof(MAT_PTR_TYPE);
    
    // int ThreadNum_short3 = 4 * WARP_SIZE;
    // int BlockNum_short3 = (short_row + ThreadNum_short3 - 1) / ThreadNum_short3;

    int *dshort3_cid;
    MAT_PTR_TYPE *dshort3_rpt;
    MAT_VAL_TYPE *dshort3_val;

    cudaMalloc((void **)&dshort3_val, sizeof(MAT_VAL_TYPE) * nnz_short);
    cudaMalloc((void **)&dshort3_cid, sizeof(int) * nnz_short);
    cudaMalloc((void **)&dshort3_rpt, sizeof(MAT_VAL_TYPE) * (short_row + 1));
    cudaMemcpy(dshort3_cid, short3_cid, sizeof(int) * nnz_short, cudaMemcpyHostToDevice);
    cudaMemcpy(dshort3_val, short3_val, sizeof(MAT_VAL_TYPE) * nnz_short, cudaMemcpyHostToDevice);
    cudaMemcpy(dshort3_rpt, short3_rpt, sizeof(MAT_PTR_TYPE) * (short_row + 1), cudaMemcpyHostToDevice);
    
    // int ThreadNum_long = warpNum_long * WARP_SIZE;

    // int BlockNum_segsum = (row_long + warpNum_long - 1) / warpNum_long;

    // int ThreadNum = NUM * WARP_SIZE;
    // int BlockNum = (blocknum + NUM - 1) / NUM;
    int BlockNum = (blocknum + rowloop * 4 - 1) / (rowloop * 4);

    int ThreadNum_short = warpNum_short * WARP_SIZE;
    int BlockNum_short_1 = (short_row_1 + ThreadNum_short - 1) / ThreadNum_short;
    // int BlockNum_short_13 = (short_block13 + warpNum_short - 1) / warpNum_short;
    // int BlockNum_short_34 = (short_block34 + warpNum_short * loopNum_short - 1) / (warpNum_short * loopNum_short); 
    // int BlockNum_short_22 = (short_block22 + warpNum_short - 1) / warpNum_short;

    // int BlockNum_short = BlockNum_short_1 + BlockNum_short_13 + BlockNum_short_34+ BlockNum_short_22;
    int BlockNum_short = BlockNum_short_1 + threadblock13 + threadblock34 + threadblock22;

    int offset_reg = BlockNum_long;
    int offset_short1 = offset_reg + BlockNum;
    int offset_short13 = offset_short1 + BlockNum_short_1;
    int offset_short34 = offset_short13 + threadblock13;
    int offset_short22 = offset_short34 + threadblock34;

    int BlockNum_all = BlockNum_long + BlockNum + BlockNum_short;
    int ThreadNum_all = 4 * WARP_SIZE;

    int sumBlockNum = (row_long + 3) / 4;

    // printf("1:%d, 13:%d, 34:%d, 22:%d\n", BlockNum_short_1, BlockNum_short_13, BlockNum_short_34, BlockNum_short_22);
    
    MAT_VAL_TYPE *dX_val, *dY_val;

    // init cuda data of long part
    MAT_VAL_TYPE *dlong_val, *dval_by_warp;
    MAT_PTR_TYPE *dlong_ptr_warp;
    int *dlong_cid; 
    int *drid_by_warp;

    // init cuda data of short part
    MAT_VAL_TYPE *dshort_val;
    int *dshort_cid;

    MAT_PTR_TYPE *dsuperX_ptr;
    MAT_VAL_TYPE *dsuperX_val;
    int *dshort_cid_new;

    // init cuda data of reg & irreg part
    MAT_VAL_TYPE *dreg_val, *dirreg_val;
    MAT_PTR_TYPE *dblock_ptr, *dirreg_rpt;
    int *dreg_cid, *dirreg_cid;

    cudaMalloc((void **)&dX_val, sizeof(MAT_VAL_TYPE) * colA);
    cudaMalloc((void **)&dY_val, sizeof(MAT_VAL_TYPE) * rowA);
    cudaMemcpy(dX_val, X_val, sizeof(MAT_VAL_TYPE) * colA, cudaMemcpyHostToDevice);
    cudaMemset(dY_val, 0.0, sizeof(MAT_VAL_TYPE) * rowA);

    cudaMalloc((void **)&dlong_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_long); 
    cudaMalloc((void **)&dlong_cid, sizeof(int) * fill0_nnz_long);
    cudaMalloc((void **)&drid_by_warp, sizeof(int) * warp_number);
    cudaMalloc((void **)&dval_by_warp, sizeof(MAT_VAL_TYPE) * warp_number);
    cudaMalloc((void **)&dlong_ptr_warp, sizeof(MAT_PTR_TYPE) * (row_long + 1));
    cudaMemcpy(dlong_val, long_val, sizeof(MAT_VAL_TYPE) * fill0_nnz_long, cudaMemcpyHostToDevice);
    cudaMemcpy(dlong_cid, long_cid, sizeof(int) * fill0_nnz_long, cudaMemcpyHostToDevice);
    cudaMemcpy(drid_by_warp, rid_by_warp, sizeof(int) * warp_number, cudaMemcpyHostToDevice);
    cudaMemcpy(dlong_ptr_warp, long_rpt_new, sizeof(MAT_PTR_TYPE) * (row_long + 1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&dsuperX_ptr, sizeof(MAT_PTR_TYPE) * (super_group + 1));
    cudaMalloc((void **)&dsuperX_val, sizeof(MAT_VAL_TYPE) * nnz_superX);
    cudaMalloc((void **)&dshort_cid_new, sizeof(int) * new_cid_len);
    cudaMemcpy(dsuperX_ptr, superX_ptr, sizeof(MAT_PTR_TYPE) * (super_group + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dsuperX_val, superX_val, sizeof(MAT_VAL_TYPE) * nnz_superX, cudaMemcpyHostToDevice);
    cudaMemcpy(dshort_cid_new, short_cid_new, sizeof(int) * new_cid_len, cudaMemcpyHostToDevice);

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

    cudaMalloc((void **)&dirreg_val, sizeof(MAT_VAL_TYPE) * nnz_irreg);
    cudaMalloc((void **)&dirreg_rpt, sizeof(MAT_PTR_TYPE) * (row_block + 1));
    cudaMalloc((void **)&dirreg_cid, sizeof(int) * nnz_irreg);
    cudaMemcpy(dirreg_val, irreg_val, sizeof(MAT_VAL_TYPE) * nnz_irreg, cudaMemcpyHostToDevice);
    cudaMemcpy(dirreg_rpt, irreg_rpt, sizeof(MAT_PTR_TYPE) * (row_block + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(dirreg_cid, irreg_cid, sizeof(int) * nnz_irreg, cudaMemcpyHostToDevice); 
    
    int carveout = 0;
    cudaFuncSetAttribute(dasp_spmv2<1>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<2>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    cudaFuncSetAttribute(dasp_spmv2<4>, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

    int warmup_time = 100;
    int execute_time = 1000;
    if (rowloop == 1)
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv2<1><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<1><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34);
        }
        cudaDeviceSynchronize();
        if (row_long)
        {
            for (int i = 0; i < execute_time; ++i)
            {
                longPart_sum<<<sumBlockNum, ThreadNum_all>>>(dlong_ptr_warp, dval_by_warp, dY_val, row_long);
            }
            cudaDeviceSynchronize();
        }
        gettimeofday(&t2, NULL);
    }
    else if (rowloop == 2)
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv2<2><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<2><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34);
        }
        cudaDeviceSynchronize();
        if (row_long)
        {
            for (int i = 0; i < execute_time; ++i)
            {
                longPart_sum<<<sumBlockNum, ThreadNum_all>>>(dlong_ptr_warp, dval_by_warp, dY_val, row_long);
            }
            cudaDeviceSynchronize();
        }
        gettimeofday(&t2, NULL);
    }
    else
    {
        for (int i = 0; i < warmup_time; ++i)
        {
            dasp_spmv2<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        for (int i = 0; i < execute_time; ++i)
        {    
            dasp_spmv2<4><<<BlockNum_all, ThreadNum_all>>>(dX_val, dY_val, 
                                                    dlong_val, dlong_cid, dval_by_warp, dlong_ptr_warp, row_long,
                                                    dreg_val, dreg_cid, dblock_ptr, row_block, blocknum, 
                                                    dirreg_val, dirreg_cid, dirreg_rpt,
                                                    dshort_val, dshort_cid, short_row_1, common_13, short_row_34, short_row_2, 
                                                    offset_reg, offset_short1, offset_short13, offset_short34, offset_short22,
                                                    fill0_nnz_short13, fill0_nnz_short34);
        }
        cudaDeviceSynchronize();
        if (row_long)
        {
            for (int i = 0; i < execute_time; ++i)
            {
                longPart_sum<<<sumBlockNum, ThreadNum_all>>>(dlong_ptr_warp, dval_by_warp, dY_val, row_long);
            }
            cudaDeviceSynchronize();
        }
        gettimeofday(&t2, NULL);
    }
    

    double dasp_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / execute_time; 
    double dasp_gflops = (double)((long)nnzA * 2) / (dasp_time * 1e6);
    double dasp_bandwidth1 = (double)data_X / (dasp_time * 1e6);
    double dasp_bandwidth2 = (double)data_X2 / (dasp_time * 1e6);
    printf("SpMV_X:  %8.4lf ms, %8.4lf GFlop/s, %9.4lf GB/s, %9.4lf GB/s\n", dasp_time, dasp_gflops, dasp_bandwidth1, dasp_bandwidth2);

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

    cudaFree(dshort3_cid);
    cudaFree(dshort3_rpt);
    cudaFree(dshort3_val);

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
    //     fprintf(fp_y, "%lf \n", Y_val[i]);
    // }
    // fclose(fp_y);

    // int result = verify(Y_val2, Y_val, rowA, 1);

    FILE* fout;
    fout = fopen("data/spmv_f64_record.csv", "a");
    fprintf(fout, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", filename, rowA, colA, nnzA, short_row_1, common_13, short_row_3, short_row_4, short_row_2, row_long, row_block, nnz_short, fill0_nnz_short, nnz_long, fill0_nnz_long, origin_nnz_reg, fill0_nnz_reg, nnz_irreg);
    fprintf(fout, "%lf,%d,%lld,%lf,%lf,%lf,%lf,", rate_fill0, block_longest, data_X, dasp_time, dasp_gflops, dasp_bandwidth1, dasp_bandwidth2);
    fclose(fout);

    printf("\n");

    free(short_rid_1);
    free(short_rid_2);
    free(short_rid_3);
    free(short_rid_4);
    free(long_rid);
    free(zero_rid);
    free(ridA);

    free(superX_ptr);
    free(superX_cid);
    free(superX_val);
    free(short_cid_temp);
    free(short_cid_new);

    free(rptA);
    free(long_rpt);

    free(short_val);
    free(short_cid);

    free(short3_cid);
    free(short3_rpt);
    free(short3_val);

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
