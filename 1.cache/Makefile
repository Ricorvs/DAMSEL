
CC=gcc
NVCC=nvcc
INCLUDES=

ifndef DEBUG
CFLAGS=-O2
LIB=
NVCCFLAGS+=-O2
else
CFLAGS=-O0 -g3
LIB=
NVCCFLAGS+=-O0 -g3
endif

OBJDIR = "obj"

PROGS := $(wildcard src/main_*.cu)
SRCS := $(filter-out $(PROGS), $(wildcard src/*.cu))
PROGS := $(PROGS:src/main_%.cu=%)
BINS := $(SRCS:src/%.cu=obj/%.o)

all: ${PROGS}

%: ${BINS} obj/main_%.o
	${NVCC} -lm $^ -o $@  $(NVCCFLAGS)

obj/%.o: src/%.cu
	@if [ ! -d $(OBJDIR) ]; then mkdir -p $(OBJDIR); fi
	$(NVCC) $(INCLUDES) -c $< -o $@  $(NVCCFLAGS)

cleanobj:
	-rm -f $(OBJ_CUDA) $(OBJ)

clean:
	-rm -r -f $(BINS) $(PROGS) $(OBJDIR)
