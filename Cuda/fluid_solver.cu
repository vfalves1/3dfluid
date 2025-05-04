#include "fluid_solver.h"
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <xmmintrin.h>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

__global__
void add_source_kernel(int size, float *x, float *s, float dt) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (idx < size) {
        x[idx] += dt * s[idx];
    }
}


// Add sources (density or velocity)
void add_source_cuda(int size, float *x, float *s, float dt) {

    int threadsPerBlock = 256;
    int blocksPerGrid = ceil(size / threadsPerBlock);

    add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, x, s, dt);
}

__global__ void set_bnd_kernel(int M, int N, int O, int b1,  int b2, int b3, float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
 
    if (i <= M && j <= N) {
        x[IX(i, j, 0)] = b3 * x[IX(i, j, 1)];  
        x[IX(i, j, O + 1)] = b3 * x[IX(i, j, O)];
    }
    if (j <= N && i <= O) {  
        x[IX(0, j, i)] = b1 * x[IX(1, j, i)];
        x[IX(M + 1, j, i)] = b1 * x[IX(M, j, i)];
    }
    if (i <= M && j <= O) {
        x[IX(i, 0, j)] = b2 * x[IX(i, 1, j)];
        x[IX(i, N + 1, j)] = b2 * x[IX(i, N, j)];
    }
 
    if (threadIdx.x == 0 && threadIdx.y == 0) { 
       x[IX(0, 0, 0)] =
         0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
       x[IX(M + 1, 0, 0)] =
         0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
       x[IX(0, N + 1, 0)] =
         0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
       x[IX(M + 1, N + 1, 0)] =
         0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
   }
}

// Set boundary conditions 
void set_bnd_cuda(int M, int N, int O, int b, float *d_x) {
    dim3 blockDim(32, 8);
    int size = fmaxf(M,fmaxf(N,O));
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, 
                 (size + blockDim.y - 1) / blockDim.y);

    set_bnd_kernel<<<gridDim, blockDim>>>(M, N, O, b==1?-1:1,b==2?-1:1,b==3?-1:1, d_x);
}


__device__ void atomicMax_float(float *address, float value) {
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}


__global__ 
void lin_solve_kernel_Black(int M, int N, int O, int b, float *x, float *x0, float a, float c, float *max_c){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z*2 + threadIdx.z *2 + 1 + (i+j+1)%2;
    
    __shared__ float local_max[256];
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    local_max[thread_id] = 0.0f; 

    if (i <= M && j <= N && k <= O){
        int idx = IX(i, j, k);
        float old_x = x[idx];
        x[idx] = (x0[idx] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
        float change = fabsf(x[idx] - old_x);
	local_max[thread_id] = change;
    }
    __syncthreads();
    for (int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s>>=1) {
        if (thread_id < s) {
            local_max[thread_id] = fmaxf(local_max[thread_id], local_max[thread_id + s]);
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        atomicMax_float(max_c, local_max[0]);
    }
}


__global__ 
void lin_solve_kernel_Red(int M, int N, int O, int b, float *x, float *x0, float a, float c, float *max_c){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z*2 + threadIdx.z *2 + 1 +(i+j)%2;
        
    __shared__ float local_max[256];
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    local_max[thread_id] = 0.0f; 
 
    if (i <= M && j <= N && k <= O){
        int idx = IX(i, j, k);
        float old_x = x[idx];
        x[idx] = (x0[idx] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) / c;
        float change = fabsf(x[idx] - old_x);
        local_max[thread_id] = change;
    }
    __syncthreads();
    for (int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s>>=1) {
        if (thread_id < s) {
            local_max[thread_id] = fmaxf(local_max[thread_id], local_max[thread_id + s]);
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        atomicMax_float(max_c, local_max[0]);
    } 
}

void lin_solve_cuda(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c;
    int l = 0;
    float *d_max_c;
    cudaMalloc((void**) &d_max_c, sizeof(float));

    //  Lançamento 3D  lin_solve_kernel_Red/Black 
    dim3 blockDim(32, 4, 2); 
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y,
                 (O + blockDim.z * 2 - 1) / (blockDim.z * 2));

do {
    max_c = 0.0f;
    cudaMemcpy(d_max_c, &max_c, sizeof(float), cudaMemcpyHostToDevice);

    lin_solve_kernel_Red<<<gridDim, blockDim>>>(M, N, O, b, x, x0, a, c, d_max_c);

    lin_solve_kernel_Black<<<gridDim, blockDim>>>(M, N, O, b, x, x0, a, c, d_max_c);

    cudaMemcpy(&max_c, d_max_c, sizeof(float), cudaMemcpyDeviceToHost);

    set_bnd_cuda(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);

    cudaFree(d_max_c);
}

	
// Diffusion step (uses implicit method)
void diffuse_cuda(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve_cuda(M, N, O, b, x, x0, a, 1 + 6 * a);
}

	
__global__ void advect_kernel(int M, int N, int O, float *d, float *d0,
                              float *u, float *v, float *w, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; 
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1; 
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1; 

    if (i > M || j > N || k > O) return;

    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;
    int idx = IX(i, j, k);
    float x = i - dtX * u[idx];
    float y = j - dtY * v[idx];
    float z = k - dtZ * w[idx];

    // Clamp to grid boundaries
    x = fmaxf(0.5f, fminf(x, M + 0.5f));
    y = fmaxf(0.5f, fminf(y, N + 0.5f));
    z = fmaxf(0.5f, fminf(z, O + 0.5f));

    int i0 = (int)x, i1 = i0 + 1;
    int j0 = (int)y, j1 = j0 + 1;
    int k0 = (int)z, k1 = k0 + 1;

    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    float c00 = u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)];
    float c01 = u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)];
    float c10 = u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)];
    float c11 = u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)];

    float c0 = t0 * c00 + t1 * c01;
    float c1 = t0 * c10 + t1 * c11;

    d[idx] = s0 * c0 + s1 * c1;

}


// Advection step (uses velocity field to move quantities)
void advect_cuda(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {

    dim3 blockDim(32, 4, 4);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                (N + blockDim.y - 1) / blockDim.y,  
                (O + blockDim.z - 1) / blockDim.z);

   advect_kernel<<<gridDim, blockDim>>>(M, N, O, d, d0, u, v, w, dt);

   set_bnd_cuda(M, N, O, b, d);
} 



// Kernel para calcular `div` e inicializar `p`
__global__ 
void compute_div_and_init_p(int M, int N, int O, float *u, float *v, float *w, float *p, float *div, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        div[idx] =  scale *
                   (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                    v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                    w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]);
        p[idx] = 0.0f;
    }
}


// Kernel para atualizar as velocidades `u`, `v` e `w` com base em `p`
__global__
void update_velocity(int M, int N, int O, float *u, float *v, float *w, float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int idx = IX(i, j, k);
        u[idx] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[idx] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[idx] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project_cuda(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div) {

  dim3 blockDim(32, 4, 4);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
               (N + blockDim.y - 1) / blockDim.y,  
               (O + blockDim.z - 1) / blockDim.z);

  float scale = -0.5f / MAX(M, MAX(N, O));

  compute_div_and_init_p<<<gridDim, blockDim>>>(M, N, O, u, v, w, p, div, scale);

  set_bnd_cuda(M, N, O, 0, div);
  set_bnd_cuda(M, N, O, 0, p);

  lin_solve_cuda(M, N, O, 0, p, div, 1, 6);

  update_velocity<<<gridDim, blockDim>>>(M, N, O, u, v, w, p);

  set_bnd_cuda(M, N, O, 1, v);
  set_bnd_cuda(M, N, O, 2, u);
  set_bnd_cuda(M, N, O, 3, w); 
}


// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  add_source_cuda(size, x, x0, dt);
  SWAP(x0, x); 
  diffuse_cuda(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect_cuda(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  add_source_cuda(size, u, u0, dt);
  add_source_cuda(size, v, v0, dt);
  add_source_cuda(size, w, w0, dt);
  SWAP(u0, u);
  diffuse_cuda(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse_cuda(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse_cuda(M, N, O, 3, w, w0, visc, dt);
  project_cuda(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect_cuda(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect_cuda(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect_cuda(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project_cuda(M, N, O, u, v, w, u0, v0);
}



