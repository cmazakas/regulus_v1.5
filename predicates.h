/*
Author: Ashwin Nanjappa and Cao Thanh Tung
Filename: GDelShewchukDevice.h

===============================================================================

Copyright (c) 2013, School of Computing, National University of Singapore.
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/gdel3d.html

If you use gStar4D and you like it or have comments on its usefulness etc., we
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the National University of University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

*/

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
