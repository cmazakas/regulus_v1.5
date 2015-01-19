# regulus_v1.5
Degenerate Delaunay in CUDA

Okay, this is something that is half-complete. But I'm tired of not talking about it and I'd love 
feedback while I continuously work on the code and update it.

As of now, this code is NOT Delaunay. This is currently an attempt at degenerate tetrahedralization
on the GPU, specifically CUDA. Delaunay refinement is a feature I'd ultimately like to add but that's
still far away.

The code is largely based off of gDel3d so I recommend you google it and find the paper because
a lot of what I do in regulus is based off of this.

Compiled on a 750 Ti with CUDA 6.5 and 7.0. The Makefile should be configured for the user's
device (instead of 50, use appropriate compute capability).
