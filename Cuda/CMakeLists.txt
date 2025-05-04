c# CMakeLists.txt dentro de cuda/

# Arquivos de código fonte para a versão CUDA
set(CUDA_SOURCES main.cu EventManager.cpp fluid_solver.cu)

# Criar o executável CUDA
add_executable(fluid_sim_cuda ${CUDA_SOURCES})

# Definir as opções de compilação para CUDA
target_compile_options(fluid_sim_cuda PRIVATE -Wall -O3)

# Especificar a arquitetura CUDA
target_compile_options(fluid_sim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_35>)

# Vincular com CUDA
target_link_libraries(fluid_sim_cuda PRIVATE CUDA::cudart)

# Incluir diretórios específicos, se necessário
# target_include_directories(fluid_sim_cuda PRIVATE <alguns_diretorios>)
