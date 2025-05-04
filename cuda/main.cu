#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>

#define SIZE 168

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
static float *d_dens, *d_dens_prev;

// Function to allocate simulation data
void allocate_data() {
  int size = (M + 2) * (N + 2) * (O + 2)*sizeof(float);

  cudaMalloc(&d_u,size);
  cudaMalloc(&d_v,size);
  cudaMalloc(&d_w,size);
  cudaMalloc(&d_u_prev,size);
  cudaMalloc(&d_v_prev,size);
  cudaMalloc(&d_w_prev,size); 
  cudaMalloc(&d_dens,size);
  cudaMalloc(&d_dens_prev,size);
}

// Function to clear the data (set all to zero)
void clear_data() {
  int size = (M + 2) * (N + 2) * (O + 2)*sizeof(float);

  cudaMemset(d_u,0,size);
  cudaMemset(d_v,0,size);
  cudaMemset(d_w,0,size);
  cudaMemset(d_u_prev,0,size);
  cudaMemset(d_v_prev,0,size);
  cudaMemset(d_w_prev,0,size);
  cudaMemset(d_dens,0,size);
  cudaMemset(d_dens_prev,0,size);
}

// Free allocated memory
void free_data() {
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_u_prev);
  cudaFree(d_v_prev);
  cudaFree(d_w_prev);
  cudaFree(d_dens);
  cudaFree(d_dens_prev);
}

void apply_events_cuda(const std::vector<Event> &events) {
  for (const auto &event : events) {
    int i = M / 2, j = N / 2, k = O / 2;
    int idx = IX(i, j, k);
    if (event.type == ADD_SOURCE) {
      // Apply density source at the center of the grid
      float d = event.density;
      cudaMemcpy(&d_dens[idx], &d, sizeof(float),cudaMemcpyHostToDevice);
    } else if (event.type == APPLY_FORCE) {
      // Apply forces based on the event's vector (fx, fy, fz)
      float fx = event.force.x, fy = event.force.y, fz = event.force.z;   
      cudaMemcpy(&d_u[idx], &fx, sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(&d_v[idx], &fy, sizeof(float),cudaMemcpyHostToDevice);
      cudaMemcpy(&d_w[idx], &fz, sizeof(float),cudaMemcpyHostToDevice);
    }
  }
}


// Function to sum the total density
float sum_density() {
  float total_density = 0.0f;
  int size = (M + 2) * (N + 2) * (O + 2);
  float *dens = (float *)malloc(size * sizeof(float));
  cudaMemcpy(dens, d_dens, size * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; i++) {
    total_density += dens[i];
  }
  return total_density;
}



// Simulation loop
void simulate(EventManager &eventManager, int timesteps) {
  for (int t = 0; t < timesteps; t++) {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events_cuda(events);

    // Perform the simulation steps
    vel_step(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev, d_w_prev, visc, dt);
    dens_step(M, N, O, d_dens, d_dens_prev, d_u, d_v, d_w, diff, dt);
  }
}

int main() {
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  allocate_data();
  
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}
