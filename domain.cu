#include "structures.h"

void regulus::build_domain(const int box_length) // length of Cartesian grid
{
    // cubic distribution + number of points for root tetrahedron
    num_cartesian_points = box_length * box_length * box_length;
    num_points           = num_cartesian_points + 4;

    // allocate memory for points
    points = thrust::device_malloc<point>(num_points);

    // write Cartesian values to points
    {
        thrust::host_vector<point> tmp;
        tmp.resize(num_cartesian_points);

        for (int i = 0; i < num_cartesian_points; ++i)
        { 
            tmp[i] = point(i / (box_length * box_length), (i / box_length) % box_length, i % box_length);
        }

        /*cudaMemcpy(points + 4,
                   thrust::raw_pointer_cast(tmp.data()),
                   num_cart_points * sizeof(*points),
                   cudaMemcpyHostToDevice);*/

        thrust::copy(tmp.begin(), tmp.end(), points + 4);
        cudaDeviceSynchronize();
    }

    // write root points to buffer
    const int root_edge_length = (box_length - 1) * 3;

    points[0] = point(0, 0, 0);
    points[1] = point(root_edge_length, 0, 0);
    points[2] = point(0, root_edge_length, 0);
    points[3] = point(0, 0, root_edge_length);

    
    // build root tetrahedron
    mesh = thrust::device_malloc<tetrahedron>(8 * num_cartesian_points);
    mesh[0] = tetrahedron(0, 1, 2, 3);
    
    // initially we only have 1 tetrahedron
    num_tetra = 1;

    // build adjacency relations
    adj_relations = thrust::device_malloc<adjacency_info>(8 * num_cartesian_points);

    // init adjacency relations
    {
        thrust::host_vector<adjacency_info> tmp;
        tmp.resize(8 * num_cartesian_points);

        thrust::copy(tmp.begin(), tmp.end(), adj_relations);
        cudaDeviceSynchronize();
    }
}
