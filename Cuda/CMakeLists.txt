# CMakeLists.txt inside cuda/

# Source files for the CUDA version
set(CUDA_SOURCES main.cu EventManager.cpp fluid_solver.cu)

# Create the CUDA executable
add_executable(fluid_sim_cuda ${CUDA_SOURCES})

# Define compilation options for CUDA
target_compile_options(fluid_sim_cuda PRIVATE -Wall -O3)

# Specify the CUDA architecture
target_compile_options(fluid_sim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_35>)

# Link with CUDA
target_link_libraries(fluid_sim_cuda PRIVATE CUDA::cudart)

# Include specific directories if needed
# target_include_directories(fluid_sim_cuda PRIVATE <some_directories>)
