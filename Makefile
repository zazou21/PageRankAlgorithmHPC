CXX ?= mpicxx
CXXFLAGS ?= -O2 -g -fopenmp -std=c++17

EXE ?= experiment
SRC ?= experiment.cpp pagerank_common.cpp
MPI_PROCS ?= 4

all: build

build:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE).exe

run: build
	mpiexec -n $(MPI_PROCS) ./$(EXE).exe

clean:
	rm -f $(EXE).exe
