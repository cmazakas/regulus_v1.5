#pragma once

#include "GDelShewchukDevice.h"

enum Orient
{
    OrientNeg   = -1,
    OrientZero  = +0,
    OrientPos   = +1
};

__device__ Orient detToOrient( float det )
{
    return ( det > 0 ) ? OrientPos : ( ( det < 0 ) ? OrientNeg : OrientZero );
}

int PredThreadNum       = 32 * 32;

template< typename T >
T* cuNew( int num )
{
    T* loc              = NULL;
    const size_t space  = num * sizeof( T );
    cudaMalloc( &loc, space );

    return loc;
}

void initPredicate(PredicateInfo &DPredicateInfo)
{
    DPredicateInfo.init();

    // Predicate constants
    DPredicateInfo._consts = cuNew< float >( DPredicateBoundNum );

    // Predicate arrays
    DPredicateInfo._data = cuNew< float >( PredicateTotalSize * PredThreadNum );

    // Set predicate constants
    kerInitPredicate<<< 1, 1 >>>( DPredicateInfo._consts );

    return;
}

__device__ Orient orientation
(
const float* predConsts,
const float* p0,
const float* p1,
const float* p2,
const float* p3
)
{
    float det = orient3dfast( predConsts, p0, p1, p2, p3 ); 
//printf("det = %f\n", det);
    // Need exact check
    if ( det == FLT_MAX )
    { //printf("Calling exact routine...\n");
        det = orient3dexact( predConsts, p0, p1, p2, p3 );
    }
//printf("%.00f\n", det);
    return detToOrient( det );
}
