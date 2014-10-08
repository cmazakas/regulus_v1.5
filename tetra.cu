#include "structures.h"
#include "predicates.h"

__global__
void fracture(const int n,
              const unsigned char *location_code,
                    tetrahedron *tetrahedra,
              const int *tet_index,
              const int *insertion_marked,
              const int num_tetrahedra,
              const int *pt_index)
{
    const int thread_num = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_num; tid < n; tid += blockDim.x * gridDim.x)
    {
        const unsigned char loc = location_code[tid];

        if (loc != 0)
        {
            // To fracture, we need the tetrahedron for its face data

            const tetrahedron t = tetrahedra[tet_index[tid]];

            const int faces[4][3] = { { t.v[3], t.v[2], t.v[1] },
                                      { t.v[0], t.v[2], t.v[3] },
                                      { t.v[0], t.v[3], t.v[1] },
                                      { t.v[0], t.v[1], t.v[2] }
                                    };

            tetrahedron *address = tetrahedra + tet_index[tid];
            int pos = 0;

            for (int i = 0; i < 4; ++i)
            {
                if (loc & (1 << i))
                {
                    const tetrahedron *tmp = 
                        new(address) tetrahedron(faces[i][0],
                                                 faces[i][1],
                                                 faces[i][2],
                                                 pt_index[tid]);

                    address = tetrahedra + num_tetrahedra
                            + insertion_marked[i] + pos;
                    ++pos;
                    
                    //printf("%d, %d, %d, %d\n", tmp->v[0], tmp->v[1], tmp->v[2], tmp->v[3]);
                }
            }
        }
    }
}

__global__
void build_fracture_info(const int           n,
                               int           *insertion_marked,
                               unsigned char *location_code,
                         const tetrahedron   *tetrahedra,
                         const int           *tet_index,
                         const point         *points,
                         const int           *pt_index,
                         const PredicateInfo preds)
{
    const int num_thread = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = num_thread; tid < n; tid += blockDim.x * gridDim.x)
    {
        // If a particular point-tetrahedron pair is marked,
        // calulate location info and write how many extra
        // tetrahedra will be added to the mesh

        if (insertion_marked[tid])
        {
            const tetrahedron t = tetrahedra[tet_index[tid]];

            const point a = points[t.v[0]];
            const point b = points[t.v[1]];
            const point c = points[t.v[2]];
            const point d = points[t.v[3]];

            const point p = points[pt_index[tid]];

            //a.print();
            //b.print();
            //c.print();
            //d.print();
            //p.print();

            // orientation == 1 (above half-space)
            // orientation == 0 (on half-space)

            const int orient0 = orientation(preds._consts, d.p, c.p, b.p, p.p);
            const int orient1 = orientation(preds._consts, a.p, c.p, d.p, p.p);
            const int orient2 = orientation(preds._consts, a.p, d.p, b.p, p.p);
            const int orient3 = orientation(preds._consts, a.p, b.p, c.p, p.p);

            assert(orient0 != -1); 
            assert(orient1 != -1);
            assert(orient2 != -1); 
            assert(orient3 != -1);

            insertion_marked[tid] = orient0 + orient1 + orient2 + orient3 - 1;

            location_code[tid] |= (orient0 << 0);
            location_code[tid] |= (orient1 << 1);
            location_code[tid] |= (orient2 << 2);
            location_code[tid] |= (orient3 << 3);

            printf("Number of extra tetrahedra : %d\n\tLocation code : %d\n", insertion_marked[tid], location_code[tid]);
        }
    }
}

__global__
void resolve_conflicts(const int num_buckets,
                       const int *pt_tet_hash,
                       const int *tet_index,
                             int *tets_to_fracture,
                             int *insertion_marked)
{
    const int thread_num = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_num; tid < num_buckets; tid += blockDim.x * gridDim.x)
    {
        // Iterate bucket contents

        const int list_start      = (tid > 0 ? pt_tet_hash[tid - 1] : 0);
        const int next_list_start = pt_tet_hash[tid];

        for (int i = list_start; i < next_list_start; ++i)
        {
            if (insertion_marked[list_start])
            {
                const int old = atomicCAS(tets_to_fracture + tet_index[i], 0, 1);
                if (old == 1)
                {
                    // If the previous value was already marked, unmark
                    // the bucket contents and return

                    for (int j = list_start; j < next_list_start; ++j)
                    {
                        insertion_marked[j] = 0;
                    }

                    printf("Unmarking bucket %d for insertion\n", tid);

                    return;
                }            
            }
        }
    }
}

__global__
void compare_distance(const int num_buckets,
                      const int *pt_tet_hash,
                      const int *pt_tet_dist,
                      const int *min_tet_dist,
                      const int *tet_index,
                            int *insertion_marked)
{
    const int thread_num = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_num; tid < num_buckets; tid += blockDim.x * gridDim.x)
    { 
        // Iterate bucket contents

        const int list_start      = (tid > 0 ? pt_tet_hash[tid - 1] : 0);
        const int next_list_start = pt_tet_hash[tid];

        for (int i = list_start; i < next_list_start; ++i)
        {
            // If point-tetra distance == minimum distance written
            //                            to that tetrahedra

            if (pt_tet_dist[i] == min_tet_dist[tet_index[i]])
            {
                // Mark bucket contents and leave

                for (int j = list_start; j < next_list_start; ++j)
                {
                    insertion_marked[j] = 1;
                }

                printf("bucket %d is marked for insertion!\n", tid);

                return;
            }
        }
    }
}

/*

    Some code taken directly from paper, Listing 4.2

*/

__global__
void compute_distance(const int           num_buckets,
                      const tetrahedron   *tetrahedra,
                      const int           *tet_index,
                      const int           *pt_tet_hash,
                      const point         *points,
                            int           *min_tet_dist,
                            int           *pt_tet_dist,
                      const PredicateInfo preds)
{
    const int thread_num = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_num; tid < num_buckets; tid += blockDim.x * gridDim.x)
    {
        const int list_start      = (tid > 0 ? pt_tet_hash[tid - 1] : 0);
        const int next_list_start = pt_tet_hash[tid];

        //printf("bucket %d is of length %d\n", tid, next_list_start - list_start);

        for (int i = list_start; i < next_list_start; ++i)
        {
            tetrahedron t = tetrahedra[tet_index[i]];

            point a = points[t.v[0]];
            point b = points[t.v[1]];
            point c = points[t.v[2]];
            point d = points[t.v[3]];

            point e = points[tid];

            float circumdistance = insphere(preds, a.p, b.p, c.p, d.p, e.p);
            int circumdist_as_int = __float_as_int(circumdistance);

            pt_tet_dist[i] = circumdist_as_int;
            atomicMax(min_tet_dist + tet_index[i], circumdist_as_int);

            printf("bucket %d's circumdistance wrt to tetra %d is : %.00f <=> %d\nproposed minimum value is : %d <=> %.00f for tetra %d\n\n", tid, tet_index[i], circumdistance, pt_tet_dist[i], min_tet_dist[tet_index[i]], __int_as_float(min_tet_dist[tet_index[i]]), tet_index[i]);;
        }
    }
}

void mesh::triangulate(void)
{
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // For the hashing to work, we need to have bucket contents
    // and bucket IDs. In our case, the bucket contents are the
    // indices of the tetrahedra and the bucket IDs and are the
    // indices of the points. Aligned such that each point is 
    // on or inside a tetrahedron in its bucket. Initially,
    // all points are on or inside the same root tetrahedron
    // Each array is of size n

    int *tet_index = 0, *pt_index = 0, *pt_tet_hash = 0;
    int n = num_points; // initially
    int num_buckets = 0;

    {
        int num_tet_bytes = n * sizeof(*tet_index);
        int num_pt_bytes = n * sizeof(*pt_index);

        cudaMallocManaged(&tet_index, num_tet_bytes);
        cudaMallocManaged(&pt_index, num_pt_bytes);

        cudaMemset(tet_index, 0, num_tet_bytes);

        int *tmp = 0; // to initialize pt_index with
        cudaMallocHost(&tmp, num_pt_bytes);

        for (int i = 0; i < num_points; ++i)
        {
            tmp[i] = 4 + i; // offset by 4 because of root points
        }

        cudaMemcpy(pt_index, tmp, num_pt_bytes, cudaMemcpyHostToDevice);
        cudaFreeHost(tmp);
        /*
        for (int i = 0; i < n; ++i)
        {
            printf("%d | %d\n", pt_index[i], tet_index[i]);
        }*/

        num_buckets = pt_index[n - 1] + 1;
        const int num_bucket_bytes = num_buckets * sizeof(*pt_tet_hash);
        cudaMallocManaged(&pt_tet_hash, num_bucket_bytes);

        find_boundaries<<<bpg, tpb>>>(pt_index, n, num_buckets, pt_tet_hash);
        cudaDeviceSynchronize();

        /*
        for (int i = 0; i < num_buckets; ++i)
        {
            printf("%d\n", pt_tet_hash[i]);
        }*/
    }

    // We calculate each point's distance from the circumcenter
    // of all  tetrahedra in its bucket
    
    {
        int *min_tet_dist = 0, *pt_tet_dist = 0;

        const int num_min_tet_dist_bytes = num_tetrahedra * sizeof(*min_tet_dist);
        const int num_pt_tet_dist_bytes = n * sizeof(*pt_tet_dist);

        cudaMallocManaged(&min_tet_dist, num_min_tet_dist_bytes);
        cudaMallocManaged(&pt_tet_dist, num_pt_tet_dist_bytes);

        cudaMemset(pt_tet_dist, 0, num_pt_tet_dist_bytes);

        {
            int *tmp = 0;
            cudaMallocHost(&tmp, num_min_tet_dist_bytes);
            for (int i = 0; i < num_tetrahedra; ++i)
                tmp[i] = -INT_MAX;
            cudaMemcpy(min_tet_dist, tmp, num_min_tet_dist_bytes, cudaMemcpyHostToDevice);
            cudaFreeHost(tmp);
        }

        PredicateInfo preds; // for robust routines
        initPredicate(preds);

        compute_distance<<<bpg, tpb>>>(num_buckets, tetrahedra, tet_index, pt_tet_hash, points, min_tet_dist, pt_tet_dist, preds);
        cudaDeviceSynchronize();

        // We now take each bucket and compare the distance that point is
        // from each one of its bucket contents. If a distance is found to
        // match the minimum distance written to the tetrahedron, mark
        // that bucket. Note, this creates data where insertions can collide.

        {
            int *insertion_marked = 0;
assert(num_buckets > 0);
            const int num_insertion_marked_bytes = n * sizeof(*insertion_marked);

            cudaMallocManaged(&insertion_marked, num_insertion_marked_bytes);
            cudaMemset(insertion_marked, 0, num_insertion_marked_bytes);

            compare_distance<<<bpg, tpb>>>(num_buckets, pt_tet_hash, pt_tet_dist, min_tet_dist, tet_index, insertion_marked);

            cudaDeviceSynchronize();

            int *tets_to_fracture = 0;

            cudaMallocManaged(&tets_to_fracture, num_tetrahedra * sizeof(*tets_to_fracture));
            cudaMemset(tets_to_fracture, 0, num_tetrahedra * sizeof(*tets_to_fracture));

            resolve_conflicts<<<bpg, tpb>>>(num_buckets, pt_tet_hash, tet_index, tets_to_fracture, insertion_marked);

            cudaDeviceSynchronize();

            /*for (int i = 0; i < n; ++i)
                printf("%d : %d <=> %d\n", i, insertion_marked[i], pt_index[i]);*/
            unsigned char *location_code = 0;
            cudaMallocManaged(&location_code, n * sizeof(*location_code));
            cudaMemset(location_code, 0, n * sizeof(*location_code));

            build_fracture_info<<<bpg, tpb>>>(n, insertion_marked, location_code, tetrahedra, tet_index, points, pt_index, preds);

            cudaDeviceSynchronize();

            thrust::exclusive_scan(thrust::device_ptr<int>(insertion_marked), thrust::device_ptr<int>(insertion_marked + n), thrust::device_ptr<int>(insertion_marked));

            fracture<<<bpg, tpb>>>(n, location_code, tetrahedra, tet_index, insertion_marked, num_tetrahedra, pt_index);

            cudaDeviceSynchronize();

            cudaFree(location_code);
            cudaFree(tets_to_fracture);
            cudaFree(insertion_marked);
        }

        cudaFree(pt_tet_dist);
        cudaFree(min_tet_dist);


    }

    cudaFree(pt_tet_hash);
    cudaFree(pt_index);
    cudaFree(tet_index);
}
