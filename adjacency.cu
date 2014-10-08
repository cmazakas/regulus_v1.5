#include "structures.h"



__host__ __device__
unsigned reverse_bits(const unsigned v)
{


    const unsigned c = (BitReverseTable256[v & 0xff] << 24) | 
                       (BitReverseTable256[(v >> 8) & 0xff] << 16) | 
                       (BitReverseTable256[(v >> 16) & 0xff] << 8) |
                       (BitReverseTable256[(v >> 24) & 0xff]);

    return c;
}

__device__
void link_tetrahedra(const adjacency_info &t1,
                     const adjacency_info &t2)
{
    
}
