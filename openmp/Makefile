CPP = g++ -Wall -O3 -g -fno-omit-frame-pointer -fopenmp -funroll-loops -ftree-vectorize -mavx -march=ivybridge 
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
 
all:  
	$(CPP) $(SRCS) -o fluid_sim

clean: 
	@echo Cleaning fluid_sim...
	@rm -f  fluid_sim
	@echo Done.

