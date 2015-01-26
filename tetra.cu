#include "structures.h"
#include "predicates.h"

__global__
void assert_no_flat_tetrahedra(const tetrahedron *mesh,
                               const point       *points,
                               const float       *predConsts,
                               const int          num_tetra)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_id; tid < num_tetra; tid += bpg * tpb)
    {
        const tetrahedron t = mesh[tid];

        const point a = points[t.v[0]];
        const point b = points[t.v[1]];
        const point c = points[t.v[2]];
        const point d = points[t.v[3]];

        const int ort0 = orientation(predConsts, d.p, c.p, b.p, a.p);
        const int ort1 = orientation(predConsts, a.p, c.p, d.p, b.p);
        const int ort2 = orientation(predConsts, a.p, d.p, b.p, c.p);
        const int ort3 = orientation(predConsts, a.p, b.p, c.p, d.p);

        const int ort = ort0 + ort1 + ort2 + ort3;

        if (ort == 0)
        {
            assert(false && "Flat tetrahedron found!");
        }
    }
}

__global__
void remove_vertices(const int aa_capacity,
                           int *pa,
                           int *ta,
                           int *fs,
                           int *la,
                           int *nominated)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int tid = thread_id; tid < aa_capacity; tid += bpg * tpb)
    {
        const int loc = la[tid];

        if (loc == 1 || loc == 2 || loc == 4 || loc == 8)
        {
            pa[tid] = ta[tid] = fs[tid] = la[tid] = -1;

            nominated[tid] = 0;
        }
    }
}
                     

__global__
void redistribution_cleanup(const int  parent_to_fl_num_buckets,
                            const int *parent_to_fl_bucket_starts,
                            const int *ta_to_pa_bucket_starts,
                                  int *pa,
                                  int *ta,
                                  int *fs,
                                  int *la,
                                  int *nominated)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // one thread per parent_to_fl bucket...
    for (int tid = thread_id; tid < parent_to_fl_num_buckets; tid += bpg * tpb)
    {
        // get begin/end indices.
        const int parent_to_fl_begin = (tid > 0 ? parent_to_fl_bucket_starts[tid - 1] : 0);
        const int parent_to_fl_end   = parent_to_fl_bucket_starts[tid];

        // if the fracture bucket is non-empty, this tetrahedron
        // was fractured...
        if (parent_to_fl_end - parent_to_fl_begin > 0)
        {
            const int ta_to_pa_begin = (tid > 0 ? ta_to_pa_bucket_starts[tid - 1] : 0);
            const int ta_to_pa_end   = ta_to_pa_bucket_starts[tid];

            for (int i = ta_to_pa_begin; i < ta_to_pa_end; ++i)
            {
                pa[i] = -1;
                ta[i] = -1;
                fs[i] = -1;
                la[i] = -1;

                nominated[i] = 0;
            }
        }
    }
}

__global__
void calculate_point_info(const int         num_points_per_bucket,
                          const int        *pa_starts,
                          const point      *points,
                          const tetrahedron t,
                          const float      *predConsts,
                          const int         redist_offset_begin,
                          const int         ta_id,
                                int        *pa,
                                int        *ta,
                                int        *fs,
                                int        *la)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= num_points_per_bucket)
        return;

    //printf("%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n%p\n", &num_points_per_bucket, pa_starts, points, &t, predConsts, &redist_offset_begin, &ta_id, pa, ta, fs, la);

    const point p = points[pa_starts[tid]];

    const point a = points[t.v[0]];
    const point b = points[t.v[1]];
    const point c = points[t.v[2]];
    const point d = points[t.v[3]];

    // orienation of p vs every face
    const int ort0 = orientation(predConsts, d.p, c.p, b.p, p.p); // 321
    const int ort1 = orientation(predConsts, a.p, c.p, d.p, p.p); // 023
    const int ort2 = orientation(predConsts, a.p, d.p, b.p, p.p); // 031
    const int ort3 = orientation(predConsts, a.p, b.p, c.p, p.p); // 012

    // if point is outside tetrahedron...
    if (ort0 < 0 || ort1 < 0 || ort2 < 0 || ort3 < 0)
        return;

    // write location association
    int loc = 0;

    loc |= (ort0 << 0);
    loc |= (ort1 << 1);
    loc |= (ort2 << 2);
    loc |= (ort3 << 3);

    const int fract_size = ort0 + ort1 + ort2 + ort3;

    if (fract_size == 0)
        t.print();

    assert(fract_size > 0 && fract_size <= 4);

    const int curr_write_offset = redist_offset_begin + tid;

    pa[curr_write_offset] = pa_starts[tid];
    ta[curr_write_offset] = ta_id;
    la[curr_write_offset] = loc;
    fs[curr_write_offset] = fract_size;
}

__global__
void redistribute_points(const int          parent_to_fl_num_buckets,
                         const int         *parent_to_fl_bucket_starts,
                         const int         *ta_to_pa_bucket_starts,
                         const int         *parent_to_fl_bucket_contents,
                         const int         *ta_to_pa_bucket_contents,
                         const tetrahedron *mesh,
                         const int          aa_size,
                         const int         *redist_offsets,
                         const float       *predConsts,
                         const point       *points,
                               int         *pa,
                               int         *ta,
                               int         *fs,
                               int         *la)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // one thread per bucket in parent_to_fl
    for (int tid = thread_id; tid < parent_to_fl_num_buckets; tid += bpg * tpb)
    {
        // get start/indices of parent_to_fl buckets
        const int parent_to_fl_begin = (tid > 0 ? parent_to_fl_bucket_starts[tid - 1] : 0);
        const int parent_to_fl_end   = parent_to_fl_bucket_starts[tid];

        // if bucket is empty...
        if (parent_to_fl_end - parent_to_fl_begin == 0)
            return;

        // get number of points associated with each tetrahedron
        const int ta_to_pa_begin = (tid > 0 ? ta_to_pa_bucket_starts[tid - 1] : 0);
        const int ta_to_pa_end   = ta_to_pa_bucket_starts[tid];
        const int num_points_per_bucket = ta_to_pa_end - ta_to_pa_begin;

        // assert that the bucket is not empty
        assert(num_points_per_bucket > 0);

        for (int i = parent_to_fl_begin;
             i < parent_to_fl_end;
             ++i)
        {
            // get tetrahedron in mesh
            const int ta_id = parent_to_fl_bucket_contents[i];
            const tetrahedron t = mesh[ta_id];
            
            // where point data starts...
            const int *pa_starts = ta_to_pa_bucket_contents +
                                   ta_to_pa_begin;

            // get beginning offset for writing new info
            const int redist_offset_begin = aa_size + 
                                            redist_offsets[i];

            const int blc = (num_points_per_bucket / tpb) + 
                            (num_points_per_bucket % tpb == 0 ? 0 : 1);

            calculate_point_info<<<blc , tpb>>>
                                (num_points_per_bucket,
                                 pa_starts,
                                 points,
                                 t,
                                 predConsts,
                                 redist_offset_begin,
                                 ta_id,
                                 pa,
                                 ta,
                                 fs,
                                 la);
        }
    }
}

/*__global__
void redistribute_points(const int          assoc_size,
                         const int          fract_num_buckets,
                         const int         *tetra_bucket_starts,
                         const int         *parent_to_fl_bucket_starts, 
                         const int         *fl,
                         const int         *nominated,
                         const tetrahedron *mesh,
                         const point       *points,
                         const float       *predConsts,
                               int         *pa,
                               int         *ta,
                               int         *fs,
                               int         *la)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every fracture bucket...
    for (int tid = thread_id; tid < fract_num_buckets; tid += bpg * tpb)
    {
        // bucket id = tid

        // number of points 
        const int num_points = tetra_bucket_starts[tid] -
                               (tid > 0 ? tetra_bucket_starts[tid - 1] : 0);

        // start of point indices...
        const int *pa_starts = pa + 
                               (tid > 0 ? tetra_bucket_starts[tid -1] : 0);

        // begin/end of bucket
        const int begin = (tid > 0 ? parent_to_fl_bucket_starts[tid - 1] : 0);
        const int end   = parent_to_fl_bucket_starts[tid];

        //const int off = (end - begin) * num_points;
        //printf("Proposed offsets in association arrays : %d\n", off);

        // iterate fracture buckets...
        for (int i = begin; i < end; ++i)
        {
            // get the tetrahedron
            const tetrahedron *t = mesh + fl[i];

            const int write_offset = assoc_size + begin + (i - begin) * num_points;
//printf("Will be writing new info to offset %d\n", write_offset);
            const int blc = (num_points / tpb) + num_points % tpb;
            
            calculate_point_info<<<blc, tpb>>>
                                (num_points,
                                 mesh,
                                 t,
                                 points,
                                 pa_starts,
                                 nominated,
                                 write_offset,
                                 predConsts,
                                 pa,
                                 ta,
                                 fs,
                                 la);
        }
    }
}*/

__global__
void get_redistribution_offsets(const int  parent_to_fl_num_buckets,
                                const int *ta_to_pa_bucket_starts,
                                const int *parent_to_fl_bucket_starts,
                                      int *redist_offsets)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // one thread per bucket in parent_to_fl
    for (int tid = thread_id; tid < parent_to_fl_num_buckets; tid += bpg * tpb)
    {
        // get the start/end of ta_to_pa buckets
        const int ta_to_pa_begin = (tid > 0 ? 
                                    ta_to_pa_bucket_starts[tid - 1] :
                                    0);
        const int ta_to_pa_end   = ta_to_pa_bucket_starts[tid];
    
        // number of points = difference in bucket starts...
        const int num_points_per_bucket = ta_to_pa_end -
                                          ta_to_pa_begin;

        // get start and end indices of buckets in parent_to_fl
        const int parent_to_fl_begin = (tid > 0 ?
                                    parent_to_fl_bucket_starts[tid - 1] :
                                    0);
        const int parent_to_fl_end   = parent_to_fl_bucket_starts[tid];

        // iterate to_to_fl bucket and write number of points in
        // each associated ta_to_pa bucket
        for (int i = parent_to_fl_begin; i < parent_to_fl_end; ++i)
        {
            redist_offsets[i] = num_points_per_bucket;
        }
    }
}

__global__
void fracture_tetrahedra(const int          nominated_size,
                         const int         *nominated,
                         const int         *ta,
                         const int         *pa,
                         const int         *la,
                         const int         *parent_to_fl_bucket_starts,
                         const int         *fl,
                               tetrahedron *mesh)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every nominated piece of data...
    for (int tid = thread_id; tid < nominated_size; tid += bpg * tpb)
    {
        if (nominated[tid])
        {
            // gather tetrahedron to fracture
            const int         ta_id = ta[tid];
            const tetrahedron t     = mesh[ta_id];

            const int pa_id = pa[tid];

            const int loc = la[tid];
//printf("location code = %d\n", loc);            
            const int faces[4][3] = { { 3, 2, 1 },   // face 0
                                      { 0, 2, 3 },   // face 1
                                      { 0, 3, 1 },   // face 2
                                      { 0, 1, 2 } }; // face 3

            // get start of bucket in fl 
            int fract_bucket_loc = (ta_id > 0 ? parent_to_fl_bucket_starts[ta_id -1] : 0);

            // iterate all possible faces
            for (int i = 0; i < 4; ++i)
            {
                // if point p is above the half-space...
                if (loc & (1 << i))
                {
                    const tetrahedron *tmp =
                    new(mesh + fl[fract_bucket_loc]) tetrahedron(t.v[faces[i][0]],
                                                                 t.v[faces[i][1]],
                                                                 t.v[faces[i][2]],
                                                                 pa_id);
                    ++fract_bucket_loc;
                    //tmp->print();
                }
            }    
        }
    }
}

__global__
void write_fracture_locations(const int  size,
                              const int *nominated,
                              const int *ta,
                              const int *alpha_sum,
                              const int *fs,
                              const int  num_tetra,
                              const int *beta_sum,
                                    int *fl,
                                    int *parent)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every element in nominated...
    for (int tid = thread_id; tid < size; tid += tpb * bpg)
    {
        // if the tetrahedron is marked...
        if (nominated[tid])
        {
            // parent tetrahedron id
            const int parent_id = ta[tid];

            // this is the beginning index we write to in
            // fl and parent
            const int begin = alpha_sum[tid];

            // first fracture location is current spot in mesh
            // all parent values = parent_id
            fl[begin]     = parent_id;
            parent[begin] = parent_id;

            // how many fractures do we have left?
            // simple!
            const int remaining_fractures = fs[tid] - 1;

            // get mesh offset value
            const int mesh_offset = num_tetra + beta_sum[tid];

            // fill in the rest of the addresses
            for (int j = 0; j < remaining_fractures; ++j)
            {
                const int index = begin + 1 + j;

                fl[index]     = mesh_offset + j;
                parent[index] = parent_id;
            }
        }
    }
}

template<typename T>
struct tuple_comp
{
    __host__ __device__
    bool operator()(const thrust::tuple<T, T, T, T, T> t, 
                    const thrust::tuple<T, T, T, T, T> v)
    {
        return thrust::get<0>(t) > thrust::get<0>(v); 
    }
};

template<typename T>
struct mesh_offset_op : public thrust::unary_function<thrust::tuple<T, T>, T>
{
    __host__ __device__
    T operator()(thrust::tuple<T, T> v1)
    {
        const int v = thrust::get<0>(v1);
        return (v > 0 ? v - 1 : 0) * thrust::get<1>(v1);
    }
};

template<typename T>
struct fract_bucket_op : public thrust::unary_function<thrust::tuple<T, T>, T> 
{
    __host__ __device__
    T operator()(thrust::tuple<T, T> v1)  
    {
        return thrust::get<0>(v1) * thrust::get<1>(v1);
    }
};

void get_fract_locations(associated_arrays               &aa,
                         int                             &num_tetra,
                         thrust::device_ptr<tetrahedron>  mesh,
                         thrust::device_ptr<point>        points,
                         float                           *predConsts)
{
    // we want to calculate addresses of each fracure in
    // mesh buffer

    // sort all data by tetrahedra first...
    thrust::sort_by_key(aa.ta,
                        aa.ta + aa.size,
                        thrust::make_zip_iterator(
                            thrust::make_tuple(aa.pa,
                                               aa.fs,
                                               aa.la,
                                               aa.nominated)));
    
    // fundamentally, we want all the fracture locations written
    // out into a 1d array.

    // we get writing addresses from the prefix sum of fs * nominated
    // we also get the total number of fracture addresses/locations
    // from the very last element of the sum

    // allocate output array for storing addresses
    thrust::device_vector<int> alpha_sum(aa.size, -1);

    // we also need to get the offsets to write to in the actual
    // mesh array itself
    // this is the same as (fs[i] - 1) * nominated[i], if nominated[i]
    // is 1

    // allocate array for storing mesh offsets
    thrust::device_vector<int> beta_sum(aa.size, -1);

    // perform modified inclusive scan (alpha sum)
    thrust::exclusive_scan(
        // beginning iterator
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(aa.fs, 
                                   aa.nominated)),
            fract_bucket_op<int>()),
        // ending iterator
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(aa.fs + aa.size, 
                                   aa.nominated + aa.size)),
            fract_bucket_op<int>()),
        // iterator to write to
        alpha_sum.begin());

    // but the mesh offsets for writing are (fs[i] - 1) * nominated[i]
    // need to perform a modified prefix sum
    // well, assuming fs[i] > 0 else no need to subtract 1
    thrust::exclusive_scan(
        // beginning iterator
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(aa.fs, 
                                   aa.nominated)),
            mesh_offset_op<int>()),
        // ending iterator
        thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(aa.fs + aa.size, 
                                   aa.nominated + aa.size)),
            mesh_offset_op<int>()),
        // iterator to write to
        beta_sum.begin());

    cudaDeviceSynchronize();

    /**********************************/
    /*std::cout << "Printing alpha sum" << std::endl;
    for (thrust::device_vector<int>::size_type i = 0; 
         i < alpha_sum.size(); 
         ++i)
    {
        std::cout << alpha_sum[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Printing beta sum" << std::endl;
    for (thrust::device_vector<int>::size_type i = 0; 
         i < beta_sum.size(); 
         ++i)
    {
        std::cout << beta_sum[i] << " ";
    }
    std::cout << std::endl;*/
    /**********************************/

    // get total number of fracture addresses
    const int num_addresses = alpha_sum.back() + 
                              aa.nominated[aa.size - 1] * 
                              aa.fs[aa.size - 1];

    //std::cout << "number of total addresses to write to " << num_addresses << std::endl;

    // allocate space for bucket contents
    // fl = fracture locations
    thrust::device_vector<int> fl(num_addresses, -1);

    // allocate array to keep track of original tetrahedron
    // to be associated with each fracture set
    thrust::device_vector<int> parent(fl.size(), -1);

    // launch kernel that writes addresses
    write_fracture_locations<<<bpg, tpb>>>
                            (aa.size,
                             aa.nominated.get(),
                             aa.ta.get(),
                             thrust::raw_pointer_cast(alpha_sum.data()),
                             aa.fs.get(),
                             num_tetra,
                             thrust::raw_pointer_cast(beta_sum.data()),
                             thrust::raw_pointer_cast(fl.data()),
                             thrust::raw_pointer_cast(parent.data()));

    cudaDeviceSynchronize();

    /**********************************/
    /*std::cout << "Printing fracture sites and associated tetra" << std::endl;
    for (thrust::device_vector<int>::size_type i = 0; 
         i < fl.size(); 
         ++i)
    {
        std::cout << fl[i] << ", " << parent[i] << std::endl;
    }*/
    /**********************************/

    //for (int i = 0; i < fl.size(); ++i)
        //std::cout << fl[i] << " : " << parent[i] << std::endl;

    // we now want to hash fracture locations by the id of the 
    // original tetrahedron
    hash_table parent_to_fl_table(fl.size(),
                                  parent.back() + 1,
                                  fl.data(),
                                  parent.data());

    // want to also hash points by tetrahedron id
    hash_table ta_to_pa_table(aa.size,
                              aa.ta[aa.size - 1] + 1,
                              aa.pa,
                              aa.ta);

    // build tables
    parent_to_fl_table.build_table();
    ta_to_pa_table.build_table();

    /**********************************/
    /*std::cout << "parent_to_fl_table info" << std::endl;
    parent_to_fl_table.print();
    std::cout << std::endl;

    std::cout << "ta_to_pa_table info" << std::endl;
    ta_to_pa_table.print();
    std::cout << std::endl;*/
    /**********************************/

    // allocate array to store offsets for redistribution
    thrust::device_vector<int> redist_offsets(fl.size(), -1);

    const int arr_cap = aa.capacity;

    thrust::device_ptr<int> iter;
    int new_size = -1;
    
    fracture_tetrahedra<<<bpg, tpb>>>
                       (aa.size,
                        aa.nominated.get(),
                        aa.ta.get(),
                        aa.pa.get(),
                        aa.la.get(),
                        parent_to_fl_table.bucket_starts.get(),
                        thrust::raw_pointer_cast(fl.data()),
                        mesh.get());

    get_redistribution_offsets<<<bpg, tpb>>>
                              (parent_to_fl_table.num_buckets,
                               ta_to_pa_table.bucket_starts.get(),
                               parent_to_fl_table.bucket_starts.get(),
                               thrust::raw_pointer_cast(redist_offsets.data()));

    thrust::exclusive_scan(redist_offsets.begin(),
                           redist_offsets.end(),
                           redist_offsets.begin());
    cudaDeviceSynchronize();
    /**********************************/
    /*std::cout << "Redistribution offsets :" << std::endl;
    for (thrust::device_vector<int>::size_type i = 0;
         i < redist_offsets.size();
         ++i)
    {
        std::cout << redist_offsets[i] << std::endl;
    }*/
    /**********************************/

    redistribute_points<<<bpg, tpb>>>
                       (parent_to_fl_table.num_buckets,
                        parent_to_fl_table.bucket_starts.get(),
                        ta_to_pa_table.bucket_starts.get(),
                        parent_to_fl_table.bucket_contents.get(),
                        ta_to_pa_table.bucket_contents.get(),
                        mesh.get(),
                        aa.size,
                        thrust::raw_pointer_cast(redist_offsets.data()),
                        predConsts,
                        points.get(),
                        aa.pa.get(),
                        aa.ta.get(),
                        aa.fs.get(),
                        aa.la.get());

    redistribution_cleanup<<<bpg, tpb>>>
                          (parent_to_fl_table.num_buckets,
                           parent_to_fl_table.bucket_starts.get(),
                           ta_to_pa_table.bucket_starts.get(),
                           aa.pa.get(),
                           aa.ta.get(),
                           aa.fs.get(),
                           aa.la.get(),
                           aa.nominated.get());

    remove_vertices<<<bpg, tpb>>>
                   (arr_cap,
                    aa.pa.get(),
                    aa.ta.get(),
                    aa.fs.get(),
                    aa.la.get(),
                    aa.nominated.get());

    thrust::sort(thrust::make_zip_iterator(
                    thrust::make_tuple(aa.ta,
                                       aa.pa,
                                       aa.fs,
                                       aa.la,
                                       aa.nominated)),
                 thrust::make_zip_iterator(
                    thrust::make_tuple(aa.ta        + arr_cap,
                                       aa.pa        + arr_cap,
                                       aa.fs        + arr_cap,
                                       aa.la        + arr_cap,
                                       aa.nominated + arr_cap)),
                 tuple_comp<int>());

    cudaDeviceSynchronize();    

    iter = thrust::find(aa.ta, aa.ta + arr_cap, -1);

    cudaDeviceSynchronize();    

    new_size = thrust::distance(aa.ta, iter); 

    cudaDeviceSynchronize();    

    const int new_num_tetra =  beta_sum.back() + 
                               beta_sum.back() * 
                               aa.nominated[aa.size - 1];

    num_tetra += new_num_tetra;

    aa.resize(new_size);

    //aa.print_with_nominated();
/*
    thrust::sort_by_key(aa.nominated,
                        aa.nominated + aa.size,
                        thrust::make_zip_iterator(
                            thrust::make_tuple(aa.pa,
                                               aa.ta,
                                               aa.fs,
                                               aa.la)));

    cudaDeviceSynchronize();


    remove_points<<<bpg, tpb>>>
                 (aa.capacity,
                  nominated,
                  pa,
                  ta,
                  fs,
                  la);

    cudaDeviceSynchronize();
std::cout << iter.get() << std::endl;
    iter     = thrust::find(aa.ta, aa.ta + arr_cap, -1);

    cudaDeviceSynchronize();
std::cout << iter.get() << std::endl;
    new_size = thrust::distance(aa.ta, iter); 

    cudaDeviceSynchronize();
    
    aa.resize(new_size);
    std::cout << "Finally, the new size is : " << new_size << std::endl;
    
    aa.print_with_nominated();
*/
}

__global__
void resolve_conflicts(const int  num_buckets,
                       const int *pa_to_ta_bucket_starts,
                       const int *ta,
                             int *flagged_tet,
                             int *nominated)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every bucket...
    for (int tid = thread_id; tid < num_buckets; tid += bpg * tpb)
    {
        // get begin/end indices of bucket in pa/ta
        const int begin = (tid > 0 ? pa_to_ta_bucket_starts[tid - 1] : 0);
        const int end   = pa_to_ta_bucket_starts[tid];

        // if bucket is empty...
        if (begin == end)
            return;

        // if the bucket is not nominated...
        if (nominated[begin] == 0)
            return;

        // iterate bucket
        for (int i = begin; i < end; ++i)
        {
            // bucket content
            const int mesh_index = ta[i];

            // check if this tetrahedron is already flagged
            const int old = atomicCAS(flagged_tet + mesh_index, // address
                                      0, // comparator
                                      1);
                                      //flagged_tet[mesh_index] + 1); // increment value by 1

            // if tetrahedron was already flagged...
            if (old > 0)
            {
                // de-nominate the bucket in pa
                /*memset(nominated + begin, 
                       0,
                       (end - begin) * sizeof(*nominated));*/

                // undo increments made by bucket traversal
                for (int j = begin; j < end; ++j)
                {
                    //atomicSub(flagged_tet + ta[j], 1);
                    atomicSub(nominated + j, 1);
                }

                // can end function early
                return;
            }
        }
    }
}

__global__
void compare_distance(const int  num_buckets,
                      const int *pa_to_ta_bucket_starts,
                      const int *ta,
                      const int *min_tet_dist,
                      const int *pa_ta_dist,
                            int *nominated)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every point bucket...
    for (int tid = thread_id; tid < num_buckets; tid += bpg * tpb)
    {
        // get begin/end indices of bucket in pa/ta
        const int begin = (tid > 0 ? pa_to_ta_bucket_starts[tid - 1] : 0);
        const int end   = pa_to_ta_bucket_starts[tid];

        if (begin == end)
            return;

        // iterate point bucket
        for (int i = begin; i < end; ++i)
        {
            // minimum distance at particular tetrahedron
            const int min_dist = min_tet_dist[ta[i]];

            // point's circumdistance
            const int pa_dist = pa_ta_dist[i];

            // if distances match...
            if (pa_dist == min_dist)
            {
                // mark the section of pa for insertion
                for (int j = begin; j < end; ++j)
                {
                    //nominated[j] = 1;
                    atomicAdd(nominated + j, 1);
                }

                // end function early
                return;
            }
        }
    }
}

__global__
void compute_distance(const int           size,
                      const int          *ta,
                      const tetrahedron  *mesh,
                      const point        *points,
                      const int          *pa,
                      const PredicateInfo preds,
                            int          *pa_ta_dist,
                            int          *min_tet_dist)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every element in pa...
    for (int tid = thread_id; tid < size; tid += bpg * tpb)
    {
        // calculate circumdistance

        // read tetrahedron
        const tetrahedron t = mesh[ta[tid]];

        // read points of tetrahedron
        const point a = points[t.v[0]];
        const point b = points[t.v[1]];
        const point c = points[t.v[2]];
        const point d = points[t.v[3]]; 

        // read point
        const point p = points[pa[tid]];

        // calculate circumdistance
        const float flt_circumdist = insphere(preds,
                                              a.p,
                                              b.p,
                                              c.p,
                                              d.p,
                                              p.p);
        
        // store circumdistance as an integer
        const int int_circumdist = __float_as_int(flt_circumdist);

        pa_ta_dist[tid] = int_circumdist;

        // store closest distance value
        if (int_circumdist != 0)
            atomicMax(min_tet_dist + ta[tid], int_circumdist);

        // printing information
        //printf("pa : %d, ta : %d => distance = %.00f, closest = %.00f\n", pa[tid], ta[tid], __int_as_float(pa_ta_dist[tid]), __int_as_float(min_tet_dist[ta[tid]]));
    }
}

void nominate_points(const PredicateInfo     &preds,
                     const int                num_tetra,
                     const thrust::device_ptr<tetrahedron>       mesh,
                     const thrust::device_ptr<point>             points,
                           associated_arrays &aa)
{
    // want to calculate the circumdistance for each
    // element of pa/ta as criteria for point insertion

    // first, sort everything by pa
    thrust::sort_by_key(aa.pa,
                        aa.pa + aa.size,
                        thrust::make_zip_iterator(
                            thrust::make_tuple(aa.ta,
                                               aa.fs,
                                               aa.la,
                                               aa.nominated)));
    cudaDeviceSynchronize();

    // write min circumdistance for each tetrahedron

    // relies on use of __float_as_int for atomics

    // store pa-ta distances
    thrust::device_vector<int> pa_ta_dist(aa.size, -1);

    // store minimum distance per tetrahedron
    thrust::device_vector<int> min_tet_dist(num_tetra, -INT_MAX);

    // use bucket hashing...
    hash_table pa_to_ta_table(aa.size,                // number of keys
                              aa.pa[aa.size - 1] + 1, // number of buckets
                              aa.ta,                     // bucket contents
                              aa.pa);                    // which bucket

    // build the table
    pa_to_ta_table.build_table();

    // keep track of which tetraheddra are flagged for insertion
    thrust::device_vector<int> flagged_tet(num_tetra, 0);

    // kernel calls
    compute_distance<<<bpg, tpb>>>
                    (aa.size,
                     aa.ta.get(),
                     mesh.get(),
                     points.get(),
                     aa.pa.get(),
                     preds,
                     thrust::raw_pointer_cast(pa_ta_dist.data()),
                     thrust::raw_pointer_cast(min_tet_dist.data()));

    compare_distance<<<bpg, tpb>>>
                    (pa_to_ta_table.num_buckets,
                     pa_to_ta_table.bucket_starts.get(),
                     aa.ta.get(),
                     thrust::raw_pointer_cast(min_tet_dist.data()),
                     thrust::raw_pointer_cast(pa_ta_dist.data()),
                     aa.nominated.get());

    resolve_conflicts<<<bpg, tpb>>>
                     (pa_to_ta_table.num_buckets,
                      pa_to_ta_table.bucket_starts.get(),
                      aa.ta.get(),
                      thrust::raw_pointer_cast(flagged_tet.data()),
                      aa.nominated.get());

    cudaDeviceSynchronize();
}

__global__
void init_la_and_fs(const int          size,
                    const int         *ta,
                    const tetrahedron *mesh,
                    const point       *points,
                    const int         *pa,
                    const float       *predConsts,
                          int         *la,
                          int         *fs)
{
    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // for every element in pa, ta, fs, la
    for (int tid = thread_id; tid < size; tid += bpg * tpb)
    {
        // read in tetrahedron
        const tetrahedron t = mesh[ta[tid]];

        // read in points
        const point a = points[t.v[0]];
        const point b = points[t.v[1]];
        const point c = points[t.v[2]];
        const point d = points[t.v[3]];

        // read point
        const point p = points[pa[tid]];

        // orienation of p vs every face
        const int ort0 = orientation(predConsts, d.p, c.p, b.p, p.p); // 321
        const int ort1 = orientation(predConsts, a.p, c.p, d.p, p.p); // 023
        const int ort2 = orientation(predConsts, a.p, d.p, b.p, p.p); // 031
        const int ort3 = orientation(predConsts, a.p, b.p, c.p, p.p); // 012

        assert(ort0 != -1);
        assert(ort1 != -1);
        assert(ort2 != -1);
        assert(ort3 != -1);
    
        // write location association
        int x = 0;

        x |= (ort0 << 0);
        x |= (ort1 << 1);
        x |= (ort2 << 2);
        x |= (ort3 << 3);

        la[tid] = x;

        // fracture size = sum of orientations 
        // 4 for 1-to-4, 3 for 1-to-3, 2 for 1-to-2
        fs[tid] = ort0 + ort1 + ort2 + ort3;
    }
}

void regulus::triangulate(void)
{
    // Build initial arrays
    associated_arrays aa(num_cartesian_points);

    // Everything initially relates to tetrahedron 0
    thrust::fill(aa.ta, aa.ta + aa.size, 0);

    // Initialize pa
    thrust::sequence(thrust::device, aa.pa, aa.pa + aa.size, 4);

    // Build predicate data
    PredicateInfo preds;
    initPredicate(preds);

    // Initialize the data of fs and la
    init_la_and_fs<<<bpg, tpb>>>
                  (aa.size,
                   aa.ta.get(),
                   mesh.get(),
                   points.get(),
                   aa.pa.get(),
                   preds._consts,
                   aa.la.get(),
                   aa.fs.get());

    cudaDeviceSynchronize();

    int iteration = 0;

    while (aa.pa[0] != -1)
    {
        //std::cout << "Iteration " << ++iteration << std::endl;

        // Call routine to nominate points for insertion
        nominate_points(preds,
                        num_tetra,
                        mesh,
                        points,
                        aa);

        // Fracture tetrahedra in mesh
        get_fract_locations(aa,
                            num_tetra,
                            mesh,
                            points,
                            preds._consts);
    }

    assert_no_flat_tetrahedra<<<bpg, tpb>>>
                             (mesh.get(),
                              points.get(),
                              preds._consts,
                              num_tetra);
    cudaDeviceSynchronize();

    // Print out the final mesh
    /*for (int i = 0; i < num_tetra; ++i)
    {
        const tetrahedron t = mesh[i];
        t.print();
    }*/
}
