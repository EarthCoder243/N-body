/*
Force kernels. Every backend computes accelerations for particles against all n particles.
Integrator and MPI code call computeForces() without caring which backend runs.
*/

#ifndef NBODY_FORCES_H
#define NBODY_FORCES_H

#include <string>
#include <vector>
#include "particles.h"
 
const int BACKEND_SEQ = 0;
const int BACKEND_OMP = 1;
const int BACKEND_CUDA = 2;

struct KernelResources
{
    int backend = BACKEND_OMP;
    std::vector<float> x, y, z, m;
};

int parseBackend(const std::string& name); //"seq", "omp", "cuda", -1 if unknown
const char* backendName(int backend);
bool initializeKernel(KernelResources& resources, int backend);
void releaseKernel(KernelResources& resources);
void computeForces(KernelResources& resources, const Vec4* pos, int n, int begin, int end,
                   Vec4* acc, const SimParams& params);

void forcesSequential(const Vec4* pos, int n, int begin, int end, Vec4* acc, const SimParams& params);
void forcesOpenMP(KernelResources& resources, const Vec4* pos, int n, int begin, int end,
                   Vec4* acc, const SimParams& params);

#endif
