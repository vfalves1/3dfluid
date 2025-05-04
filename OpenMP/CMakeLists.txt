# CMakeLists.txt dentro de openmp/

# Arquivos de código fonte para a versão OpenMP
set(OPENMP_SOURCES main.cpp EventManager.cpp fluid_solver.cpp)

# Criar o executável OpenMP
add_executable(fluid_sim_omp ${OPENMP_SOURCES})

# Definir as opções de compilação para OpenMP
target_compile_options(fluid_sim_omp PRIVATE -Wall -O3)

# Incluir o suporte a OpenMP
if(OpenMP_CXX_FLAGS)
    target_compile_options(fluid_sim_omp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(fluid_sim_omp PRIVATE OpenMP::OpenMP_CXX)
endif()
