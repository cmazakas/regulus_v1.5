#include "structures.h"

__global__
void peanohash(point *p, unsigned long *table, int num_points);

void mesh::create_input(const int box_length)
{
    num_points = box_length * box_length * box_length;
    --num_points; // (0, 0, 0) is repeated (it's a root point)
    num_tetrahedra = 1; // initial root tetrahedron

    const int root_edge_length = 3 * (box_length - 1);

    {
        const int pad = num_points + 4;

        point *tmp = 0;
        cudaMallocHost(&tmp, pad * sizeof(*tmp));

        // root points come first

        new(tmp + 0) point(0, 0, 0);
        new(tmp + 1) point(root_edge_length, 0, 0);
        new(tmp + 2) point(0, root_edge_length, 0);
        new(tmp + 3) point(0, 0, root_edge_length);

        for (int i = 1; i < (num_points + 1); ++i) 
        {
            new(tmp + 3 + i) point(i / (box_length * box_length), (i / box_length) % box_length, i % box_length);
            //tmp[3 + i].print();
        }

        cudaMalloc(&points, pad * sizeof(*points));
        cudaMemcpy(points, tmp, pad * sizeof(*points), cudaMemcpyHostToDevice);

        cudaFreeHost(tmp);
    }

    // Build root tetra

    tetrahedron t(0, 1, 2, 3);
    cudaMalloc(&tetrahedra, 8 * num_points * sizeof(*tetrahedra));
    cudaMemcpy(tetrahedra, &t, sizeof(t), cudaMemcpyHostToDevice);

    // Build adjacency relation table!
    // These structures align perfectly with the mesh buffer
    // and contain the adjacency information for the mesh
    // Normally, this data is kept in the tetra structure
    // but for performance reasons, it's a good idea to split
    // them up as in gFlip

    // Null-neighbours are noted by the index -1

    adjacency_relations= 0;
    cudaMalloc(&adjacency_relations, 8 * num_points * sizeof(*adjacency_relations));

    {
        adjacency_info *tmp = 0;
        cudaMallocHost(&tmp, 8 * num_points * sizeof(*tmp));
        for (int i = 0; i < 8 * num_points; ++i)
            new(tmp + i) adjacency_info();
        cudaMemcpy(adjacency_relations, tmp, 8 * num_points * sizeof(*adjacency_relations), cudaMemcpyHostToDevice);
        cudaFreeHost(tmp);
    }
};

void mesh::sort_by_peanokey(void)
{
    unsigned long *table = 0;

    cudaMalloc(&table, num_points * sizeof(*table));
    
    peanohash<<<bpg, tpb>>>(points, table, num_points);
    thrust::sort_by_key(thrust::device_ptr<unsigned long>(table), thrust::device_ptr<unsigned long>(table + num_points), thrust::device_ptr<point>(points + 4));
    cudaDeviceSynchronize();
/*
    {
        printf("\nSorted point set : \n");
        point *tmp = 0;
        cudaMallocHost(&tmp, (num_points + 4) * sizeof(*tmp));
        cudaMemcpy(tmp, points, (num_points + 4) * sizeof(*tmp), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_points + 4; ++i)
            tmp[i].print();
        cudaFreeHost(tmp);
    }
*/
    cudaFree(table);
}

__global__
void peanohash(point *p, unsigned long *table, const int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        point tmp = p[4 + i];

        table[i] =  peano_hilbert_key(tmp.x, tmp.y, tmp.z, 8 * sizeof(real));
    }
}
