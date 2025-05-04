CC = g++
NVCC = nvcc
CFLAGS = -Wall -O3
CUDAFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
SRC = main.cu EventManager.cpp fluid_solver.cu
OBJ = main.o EventManager.o fluid_solver.o
EXEC = fluid_sim

CUDA_PATH = /share/apps/cuda/11.3.1
LIBRARY_PATH = $(CUDA_PATH)/lib64

# Defina o alvo padrão: compilar e linkar com CUDA
all:  $(OBJ)
	$(NVCC) $(OBJ) -o $(EXEC) -L$(LIBRARY_PATH) -lcudart

# Compilando o arquivo main.cu com NVCC para gerar main.o
main.o: main.cu
	$(NVCC) $(CUDAFLAGS) -c main.cu -o main.o

# Compilando o EventManager.cpp com g++
EventManager.o: EventManager.cpp
	$(CC) $(CFLAGS) -c EventManager.cpp -o EventManager.o

# Compilando o fluid_solver.cu com NVCC para gerar fluid_solver.o
fluid_solver.o: fluid_solver.cu
	$(NVCC) $(CUDAFLAGS) -c fluid_solver.cu -o fluid_solver.o

clean: 
	@echo Cleaning fluid_sim...
	@rm -f  $(OBJ) $(EXEC)
	@echo Done.
