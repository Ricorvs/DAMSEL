
CC=gcc
NVCC=nvcc
INCLUDES=

ifndef DEBUG
CFLAGS=-O2
LIB=
NVCCFLAGS+=-O2 -lm
else
CFLAGS=-O0 -g3
LIB=
NVCCFLAGS+=-O0 -g3 -lm
endif

OBJDIR = "obj"

all: arithmetic matmult

matmult: matmult.cu
	${NVCC} -lm $^ -o $@  $(NVCCFLAGS)

arithmetic: arithmetic.cu
	${NVCC} -lm $^ -o $@  $(NVCCFLAGS)

cleanobj:
	-rm -f arithmetic matmult

clean:
	-rm -r -f arithmetic matmult
