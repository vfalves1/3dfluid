# CMakeLists.txt inside openmp/

# Source files for the OpenMP version
set(OPENMP_SOURCES main.cpp EventManager.cpp fluid_solver.cpp)

# Create the OpenMP executable
add_executable(fluid_sim_omp ${OPENMP_SOURCES})

# Define compilation options for OpenMP
target_compile_options(fluid_sim_omp PRIVATE -Wall -O3)

# Include OpenMP support
if(OpenMP_CXX_FLAGS)
    target_compile_options(fluid_sim_omp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(fluid_sim_omp PRIVATE OpenMP::OpenMP_CXX)
endif()
