MKFILE		= Makefile

NVCC		= nvcc -O3 -lstdc++ -lcudadevrt -rdc=true -gencode arch=compute_50,code=sm_50 -maxrregcount 32

CUSOURCE	= main.cu domain.cu peano.cu assoc.cu hash.cu tetra.cu
CUHEADER	= structures.h predicates.h GDelShewchukDevice.h 

CUOBJECTS	= ${CUSOURCE:.cu=.o}
EXECBIN		= regulus

all	: ${EXECBIN}

${EXECBIN} : ${CUOBJECTS}
	${NVCC} -o $@ ${CUOBJECTS}

%.o : %.cu

	${NVCC} -c $<

clean :
	- rm ${CUOBJECTS} ${EXECBIN}

again :
	${MAKE} clean all
