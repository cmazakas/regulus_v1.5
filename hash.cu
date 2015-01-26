#include "structures.h"

__global__
void find_boundaries(const int  num_keys,
                     const int  num_buckets,
                     const int *which_bucket,
                           int *bucket_starts)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_id; tid < num_keys; tid += bpg * tpb)
    {
        // get start and end of each bucket
        const int begin = (tid > 0 ? which_bucket[tid - 1] : 0);
        const int end   = which_bucket[tid];

        // if bucket has length > 0...
        if (begin != end)
        {
            for (int i = begin; i < end; ++i)
            {
                // sets bucket starts value to index of bucket
                bucket_starts[i] = tid;
            }
        }

        // last thread writes number of elements to the rest
        // of the array
        if (tid == num_keys - 1)
        {
            for (int i = end; i < num_buckets; ++i)
            {
                bucket_starts[i] = num_keys;
            }
        }
    }
}

hash_table::hash_table(const int                     Num_keys,
                       const int                     Num_buckets,
                       const thrust::device_ptr<int> Bucket_contents,
                       const thrust::device_ptr<int> Which_bucket)
{
    num_keys        = Num_keys;
    num_buckets     = Num_buckets;
    bucket_contents = Bucket_contents;
    which_bucket    = Which_bucket;

    bucket_starts = thrust::device_malloc<int>(num_buckets);
}

hash_table::~hash_table(void)
{
    thrust::device_free(bucket_starts);
}

void hash_table::build_table(void)
{
    find_boundaries<<<bpg, tpb>>>
                   (num_keys,
                    num_buckets,
                    which_bucket.get(),
                    bucket_starts.get());

    cudaDeviceSynchronize();
}

void hash_table::print(void)
{
    std::cout << "number of keys : " << num_keys << std::endl;
    std::cout << "number of buckets : " << num_buckets << std::endl;

    std::cout << "bucket_starts :" << std::endl;
    for (int i = 0; i < num_buckets; ++i)
    {
        std::cout << "contents of bucket " << i << " :" << std::endl;

        const int begin = (i > 0 ? bucket_starts[i - 1] : 0);
        const int end   = bucket_starts[i];

        std::cout << "end - begin = " << (end - begin) << std::endl;

        for (int j = begin; j < end; ++j)
        {
            std::cout << bucket_contents[j] << std::endl;
        }
    }
}
