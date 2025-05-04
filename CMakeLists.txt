cmake_minimum_required(VERSION 3.5)
project(fluid_simulation)

# Definir a versão do C++ padrão
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Habilitar o suporte a CUDA
find_package(CUDA REQUIRED)

# Habilitar OpenMP (se disponível)
find_package(OpenMP REQUIRED)

# Incluir subdiretórios para cada versão
add_subdirectory(cuda)
add_subdirectory(openmp)
