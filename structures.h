#include <stdio.h>
#include <assert.h>

#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/fill.h>

const int bpg = 512;
const int tpb = 256;

/*
    This is the basic point structure for Regulus.

    The use of a union between an array and a struct
    is a tad sketchy. I do a cross-platform check with
    the constructor. If it fails then that's all my fault.

    Only works if compiler keeps size at 12 bytes
*/

struct point
{
    union
    {
        float p[3];

        struct
        {
            float x, y, z;
        };
    };

    __host__ __device__
    point(void)
    {
        assert(&x == p && &y == p + 1 && &z == p + 2);

        x = y = z = 0;
    };

    __host__ __device__
    point(float a, float b, float c)
    {
        assert((&x == p) && (&y == p + 1) && (&z == p + 2));

        x = a;
        y = b;
        z = c;
    };

    __host__ __device__
    void print(void) const
    {
        printf("%.00f, %.00f, %.00f\n", x, y, z);
    }
};

struct tetrahedron
{
    int v[4];

    __host__ __device__
    tetrahedron(void)
    {
        v[0] = v[1] = v[2] = v[3] = -1;
    };

    __host__ __device__
    tetrahedron(const int a, 
                const int b, 
                const int c, 
                const int d)
    {
        v[0] = a;
        v[1] = b;
        v[2] = c;
        v[3] = d;
    };

    __host__ __device__
    void print(void) const
    {
        printf("%d, %d, %d, %d\n", v[0], v[1], v[2], v[3]);
    }
};

struct adjacency_info
{
    int ngb[4];

    __host__ __device__
    adjacency_info(void)
    {
        ngb[0] = ngb[1] = ngb[2] = ngb[3] = -1;
    }; 

    __host__ __device__
    adjacency_info(const int a, 
                   const int b, 
                   const int c, 
                   const int d)
    {
        ngb[0] = a;
        ngb[1] = b;
        ngb[2] = c;
        ngb[3] = d;
    };
};

struct associated_arrays
{
    int size, capacity;

    // These are the 4 main arrays of regulus
    // pa[i] = point in points array
    // ta[i] = id of tetra containing pa[i]
    // la[i] = location code of pa[i] relative to ta[i]
    // fs[i] = fracture size of pa[i] wrt to ta[i]

    thrust::device_ptr<int> pa, ta, fs, la, nominated;

    associated_arrays(const int num_cartesian_points);
    ~associated_arrays(void);

    void resize(const int N);
    void print(void) const;
    void print_with_nominated(void) const;
};

struct hash_table
{
    int num_keys,
        num_buckets;

    //const int *bucket_contents,
              //*which_bucket;
          //int *bucket_starts;

    thrust::device_ptr<int> bucket_contents,
                            which_bucket,
                            bucket_starts;

    hash_table(const int                     Num_keys,
               const int                     Num_buckets,
               const thrust::device_ptr<int> Bucket_contents,
               const thrust::device_ptr<int> Which_bucket);

    ~hash_table(void);

    void build_table(void);
};

struct regulus
{
    thrust::device_ptr<point>          points;
    thrust::device_ptr<tetrahedron>    mesh;
    thrust::device_ptr<adjacency_info> adj_relations;

    int num_cartesian_points,
        num_points,
        num_tetra;

    void build_domain(const int box_length);
    void sort_domain_by_peanokey(void);
    void triangulate(void);

    regulus(const int box_length)
    {
        build_domain(box_length);
        sort_domain_by_peanokey();
    };

    ~regulus(void)
    {
        thrust::device_free(points);
        thrust::device_free(mesh);
        thrust::device_free(adj_relations);
    };
};
