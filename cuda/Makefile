CC = g++
NVCC = nvcc
CFLAGS = -Wall -O3
CUDAFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets
SRC = main.cu EventManager.cpp fluid_solver.cu
OBJ = main.o EventManager.o fluid_solver.o
EXEC = fluid_sim

CPP_seq = g++ -Wall -O3 -g -fno-omit-frame-pointer -funroll-loops -ftree-vectorize 
SRC_Seq = main.cpp fluid_solver.cpp EventManager.cpp

CUDA_PATH = /share/apps/cuda/11.3.1

# Sbatch scripts
CUDA_SBATCH = cuda.sh
SEQ_SBATCH = sequential.sh


all: par seq

seq:
	$(CPP_seq) $(SRC_Seq) -o fluid_sim_seq

par: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXEC) -L$(CUDA_PATH)/lib64 -lcudart

main.o: main.cu
	$(NVCC) $(CUDAFLAGS) -c main.cu -o main.o

EventManager.o: EventManager.cpp
	$(CC) $(CFLAGS) -c EventManager.cpp -o EventManager.o

fluid_solver.o: fluid_solver.cu
	$(NVCC) $(CUDAFLAGS) -c fluid_solver.cu -o fluid_solver.o

clean: clean_seq clean_par
	@echo All clean.

clean_seq:
	@echo Cleaning fluid_sim_seq...
	@rm -f fluid_sim_seq
	@echo Done.

clean_par:
	@echo Cleaning fluid_sim...
	@rm -f  $(OBJ) $(EXEC)
	@echo Done.

run:
	sbatch $(CUDA_SBATCH)

runseq:
	sbatch $(SEQ_SBATCH)
