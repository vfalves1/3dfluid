# Fluid Simulation with Event Manager (Optimized with OpenMP and CUDA)

This project implements a fluid simulation using Jos Stam's stable fluid solver in 3D, incorporating dynamic events such as adding density sources and applying forces at specified timesteps. The simulation is optimized with OpenMP (parallelism on CPU) and CUDA (parallelism on GPU) to enhance performance, and was developed as the project for the Parallel Computing course at the University of Minho (2024/2025).

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Generating Event Data](#generating-event-data)
   - [Running the Fluid Simulation](#running-the-fluid-simulation)
5. [Example](#example)
6. [Credits](#credits)

## Project Structure

The project directory is structured as follows:
```
├── README.md            # This file
├── events.txt           # Example event file (can be generated by Python script)
├── generate_events.py   # Python script to generate events for the simulation
├── report.pdf           # Final project report
├── LICENSE              # License for the project
├── CMakeLists.txt       # CMake configuration for building the project
│
├── Cuda                 # Folder with CUDA implementation
│   ├── CMakeLists.txt   # CMake configuration for CUDA version
│   ├── EventManager.cpp # Implementation of event management (CUDA version)
│   ├── EventManager.h   # Header for event management (CUDA version)
│   ├── Makefile         # Makefile to compile CUDA simulation
│   ├── fluid_solver.cu  # CUDA implementation of the fluid solver
│   ├── fluid_solver.h   # Header for the fluid solver (CUDA version)
│   └── main.cu          # Main logic of CUDA-based simulation
│
├── OpenMP               # Folder with OpenMP implementation
│   ├── CMakeLists.txt   # CMake configuration for OpenMP version
│   ├── EventManager.cpp # Implementation of event management (OpenMP version)
│   ├── EventManager.h   # Header for event management (OpenMP version)
│   ├── Makefile         # Makefile to compile OpenMP simulation
│   ├── fluid_solver.cpp # OpenMP implementation of the fluid solver
│   ├── fluid_solver.h   # Header for the fluid solver (OpenMP version)
│   └── main.cpp         # Main logic of OpenMP-based simulation
```

## Requirements

- **C++ Compiler**: The C++ code relies on the standard C++ library.
- **Python**: The Python script generate_events.py is used to generate event data for the fluid simulation.
- **OpenMP**: Used for parallelism on the CPU.
- **CUDA**: Used for parallelism on the GPU.

### Libraries and Tools
- Python
- Standard C++ libraries for compilation (e.g., `g++` or `clang++`).
- OpenMP libraries (e.g., libomp for compiling with OpenMP)
- CUDA libraries (e.g., libcudart for compiling with CUDA)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/vfalves1/3dfluid.git
    cd 3dfluid
    ```

2. **Compile the C++ Simulation with OpenMP or CUDA**:
You can build the simulation using either the Makefile, depending on whether you want to run it on the CPU (OpenMP) or the GPU (CUDA).
1.**To compile with OpenMP**:
    ```bash
    cd OpenMP
    make
    ```

    This will generate an executable named `fluid_sim_omp`.
   
2.**To compile with  CUDA**:
    ```bash
    cd Cuda
    make
    ```

    This will generate an executable named `fluid_sim_cuda`.

## Usage

### Generating Event Data

The Python script `generate_events.py` generates a file (`events.txt`) that specifies density sources and forces applied during specific timesteps. All events are generated with normalized vectors, and the simulation will apply them at the correct timestep.

1. **Run the Python script**:
    ```bash
    python generate_events.py
    ```

    This will create an `events.txt` file with 1000 timesteps, where sources are density values, and forces are applied in positive directions along the X, Y, or Z axes.

2. **Customize the Event Generation**:
    You can modify the script to generate different events by changing the number of events, the range of timesteps, or the magnitude of forces and densities.

### Running the Fluid Simulation

Once you've generated the `events.txt` file, you can run the fluid simulation, which will read the events and apply them over the course of the simulation.

1. **Run the simulation with OpenMP**:
   -**Run with OpenMP**:
    ```bash
    ./fluid_sim_omp
    ```
    
 -**Run with Cuda**:
    ```bash
    ./fluid_sim_cuda
    ```
    The simulation will read the `events.txt` file, apply the specified sources and forces at the correct timesteps, and calculate the fluid dynamics.

3. **Simulation Output**:
    At the end of the simulation, the total density in the fluid field will be printed.

    Example output:
    ```
    Total density after 1000 timesteps: 3456.789000
    ```

## Example

### Generating Events

Here’s an example of how you can use the Python script to generate a file of events (`events.txt`):

```bash
python generate_events.py
```

This will generate an event file like the following:

```
1000
source 10 50
force 1 0 0 200
source 5 150
force 0 1 0 400
source 8 900
```

### Running the Simulation

After generating the event file, choose which version to run:

1. **OpenMP version (CPU)**:
```bash
./fluid_sim_omp
```
2. **CUDA version (GPU)**:
```bash
./fluid_sim_omp
```

Both versions will read from the same events.txt file and simulate fluid dynamics according to the specified events.

At the end, the program will print something like:
```bash
Total density after 1000 timesteps: 3456.789000
```

## Credits

This code was developed as the final project for the **Parallel Computing** course at the **University of Minho**, 2024/2025.

Authors:
- **Vitor Fernandes Alves** 
- **Diogo Gomes Rodrigues**  
- **Gonçalo Ribeiro Rodrigues** 
