#include "structures.h"

/*

    In the paper, the author uses the following variables in their
    Listing 4.1 : which_bucket, bucket_starts, num_keys, num_buckets

    pt_index = which_bucket
    pt_tet_hash = bucket_starts
    n = num_keys
    num_buckets is the same (number of buckets, duh!)

    Bucket i starts at bucket_starts[i - 1] in bucket_contents
    => bucket_contents[bucket_starts[i - 1]] is where bucket i
    resides in the physical memory

    By default, bucket 0 begins at 0 in bucket_contents
    Final element of bucket_starts stores the length of the data (n)

    This algorithm is predicated upon tet_index being sorted by pt_index
*/

__global__
void find_boundaries(const int *pt_index,
                     const int n,
                     const int num_buckets,
                           int *pt_tet_hash)
{
    const int thread_num = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_num; tid < n; tid += blockDim.x * gridDim.x)
    {
        const int prev = (tid > 0 ? pt_index[tid - 1] : 0);
        const int curr = pt_index[tid];

        // If prev != curr, we've found a boundary 

        if (prev != curr)
        {
            for (int i = prev; i < curr; ++i)
            {
                pt_tet_hash[i] = tid;
            }
        }

        // Final element 

        if (tid == n - 1)
        {
            for (int i = curr; i < num_buckets; ++i)
            {
                pt_tet_hash[i] = n;
            }
        }
    }
}

/*
__global__
void find_boundaries(const int n, // size of pa, pa.size == ta.size
                     const int nb, // number of buckets
                     const int *pa, // index of bucket per ta value
                           int *ht)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        const int prev = (i > 0 ? pa[i - 1] : 0);
        const int curr = pa[i]; 

        if (prev != curr)
        {
            for (int j = prev; j < curr; ++j)
            {
                ht[j] = i;
            }
        }

        if (i == n - 1)
        {
            for (int j = curr; j < nb; ++j)
            {
                ht[j] = n;
            }
        }       
    }   
}
*/

