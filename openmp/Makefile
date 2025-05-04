CPP = g++ -Wall -O3 -g -fno-omit-frame-pointer -fopenmp -funroll-loops -ftree-vectorize -mavx -march=ivybridge 
CPP_seq = g++ -Wall -O3 -g -fno-omit-frame-pointer -funroll-loops -ftree-vectorize -mavx -march=ivybridge
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
 
all:  seq par

seq:
	$(CPP_seq) $(SRCS) -o fluid_sim_seq

par:
	$(CPP) $(SRCS) -o fluid_sim

clean: clean_seq clean_par
	@echo All clean.

clean_seq:
	@echo Cleaning fluid_sim_seq...
	@rm -f fluid_sim_seq
	@echo Done.

clean_par:
	@echo Cleaning fluid_sim...
	@rm -f  fluid_sim
	@echo Done.

runseq:	 
	./fluid_sim_seq 

runpar: fluid_sim
	export OMP_NUM_THREADS=18;\
	./fluid_sim 
