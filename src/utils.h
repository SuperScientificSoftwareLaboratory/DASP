#include "common.h"

int BinarySearch(int *arr, int len, int target) {
	int low = 0;
	int high = len;
	int mid = 0;
	while (low <= high) {
		mid = (low + high) / 2;
		if (target < arr[mid]) high = mid - 1;
		else if (target > arr[mid]) low = mid + 1;
		else return mid;
	}
	return -1;
}

void swap_key(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

// quick sort key (child function)
int partition_key(int *key, int length, int pivot_index)
{
    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);

    for (; i < length; i++)
    {
        if (key[pivot_index + i] < pivot)
        {
            swap_key(&key[pivot_index + i], &key[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);

    return small_length;
}

// quick sort key (child function)
int partition_key_idx(int *key, int *len, int length, int pivot_index)
{
    int i = 0;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_key(&len[pivot_index], &len[pivot_index + (length - 1)]);

    for (; i < length; i++)
    {
        if (key[pivot_index + i] < pivot)
        {
            swap_key(&key[pivot_index + i], &key[small_length]);
            swap_key(&len[pivot_index + i], &len[small_length]);
            small_length++;
        }
    }

    swap_key(&key[pivot_index + length - 1], &key[small_length]);
    swap_key(&len[pivot_index + length - 1], &len[small_length]);

    return small_length;
}

// quick sort key (main function)
void quick_sort_key(int *key, int length)
{
    if (length == 0 || length == 1)
        return;

    int small_length = partition_key(key, length, 0);
    quick_sort_key(key, small_length);
    quick_sort_key(&key[small_length + 1], length - small_length - 1);
}

void quick_sort_key_idx(int *key, int *len, int length)
{
    if (length == 0 || length == 1)
        return;

    int small_length = partition_key_idx(key, len, length, 0);
    quick_sort_key_idx(key, len, small_length);
    quick_sort_key_idx(&key[small_length + 1], &len[small_length + 1], length - small_length - 1);
}

void initVec(MAT_VAL_TYPE *vec, int length)
{
    for (int i = 0; i < length; ++ i)
    {
        // vec[i] = rand() % 20 * 0.1;
        vec[i] = 1;
    }
}

#ifdef f64
__device__ __forceinline__ void mma_m8n8k4(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]):
        "d"(frag_a), "d"(frag_b)
    );
}
#endif


int get_max(int *arr, int len)
{
    int max = arr[0];
    for (int i = 1; i < len; i ++)
    {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

void count_sort(int *arr, int *idx, int len, int exp)
{
    int *temp_arr = (int *)malloc(sizeof(int) * len);
    int *temp_idx = (int *)malloc(sizeof(int) * len);
    int buckets[10] = {0};

    for (int i = 0; i < len; i ++)
    {
        buckets[(arr[i] / exp) % 10] ++;
    }

    for (int i = 1; i < 10; i ++)
    {
        buckets[i] += buckets[i - 1];
    }

    for (int i = 0; i < len; i ++)
    {
        int offset = len - (buckets[(arr[i] / exp) % 10] - 1) - 1;
        temp_arr[offset] = arr[i];
        temp_idx[offset] = idx[i];
        buckets[(arr[i] / exp) % 10] --;
    }

    for (int i = 0; i < len; i ++)
    {
        arr[i] = temp_arr[i];
        idx[i] = temp_idx[i];
    }

    free(temp_arr);
    free(temp_idx);
}

void count_sort_asce(int *arr, int *idx, int len, int exp)
{
    int *temp_arr = (int *)malloc(sizeof(int) * len);
    int *temp_idx = (int *)malloc(sizeof(int) * len);
    int buckets[10] = {0};

    for (int i = 0; i < len; i ++)
    {
        buckets[(arr[i] / exp) % 10] ++;
    }

    for (int i = 1; i < 10; i ++)
    {
        buckets[i] += buckets[i - 1];
    }

    for (int i = len - 1; i >= 0; i ++)
    {
        int offset = buckets[(arr[i] / exp) % 10] - 1;
        temp_arr[offset] = arr[i];
        temp_idx[offset] = idx[i];
        buckets[(arr[i] / exp) % 10] --;
    }

    for (int i = 0; i < len; i ++)
    {
        arr[i] = temp_arr[i];
        idx[i] = temp_idx[i];
    }

    free(temp_arr);
    free(temp_idx);
}

void radix_sort(int *arr, int *idx, int len)
{
    int max = get_max(arr, len);
    for (int exp = 1; max / exp > 0; exp *= 10)
    {
        count_sort(arr, idx, len, exp);
    }
}

void radix_sort_asce(int *arr, int *idx, int len)
{
    int max = get_max(arr, len);
    for (int exp = 1; max / exp > 0; exp *= 10)
    {
        count_sort_asce(arr, idx, len, exp);
    }
}