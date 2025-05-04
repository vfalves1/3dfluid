#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt);
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt);

#endif // FLUID_SOLVER_H
