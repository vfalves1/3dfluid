CC = g++
NVCC = nvcc
CFLAGS = -Wall -O3
CUDAFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
SRC = main.cu EventManager.cpp fluid_solver.cu
OBJ = main.o EventManager.o fluid_solver.o
EXEC = fluid_sim

CUDA_PATH = /share/apps/cuda/11.3.1
LIBRARY_PATH = $(CUDA_PATH)/lib64

# Define the default target: compile and link with CUDA
all:  $(OBJ)
	$(NVCC) $(OBJ) -o $(EXEC) -L$(LIBRARY_PATH) -lcudart

# Compile the main.cu file with NVCC to generate main.o
main.o: main.cu
	$(NVCC) $(CUDAFLAGS) -c main.cu -o main.o

# Compile EventManager.cpp with g++ to generate EventManager.o
EventManager.o: EventManager.cpp
	$(CC) $(CFLAGS) -c EventManager.cpp -o EventManager.o

# Compile fluid_solver.cu with NVCC to generate fluid_solver.o
fluid_solver.o: fluid_solver.cu
	$(NVCC) $(CUDAFLAGS) -c fluid_solver.cu -o fluid_solver.o

clean: 
	@echo Cleaning fluid_sim...
	@rm -f  $(OBJ) $(EXEC)
	@echo Done.
