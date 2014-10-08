MKFILE		= Makefile

NVCC		= nvcc -O3 -lstdc++ -rdc=true -gencode arch=compute_50,code=sm_50

CUSOURCE	= main.cu domain.cu peano.cu tetra.cu hash.cu
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
