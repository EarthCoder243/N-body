/*
Backend selection: maps "seq" | "omp" | "cuda" to a kernel and calls computeForces()
*/

#include "forces.h"
#include <cstdio>

int parseBackend(const std::string& name)
{
    if(name == "seq")
    {
        return BACKEND_SEQ;
    }
    else if(name == "omp")
    {
        return BACKEND_OMP;
    }
    else if(name == "cuda")
    {
        return BACKEND_CUDA;
    }
    return -1;
}

const char* backendName(int backend)
{
    if(backend == BACKEND_SEQ)
    {
        return "seq";
    }
    else if(backend == BACKEND_OMP)
    {
        return "omp";
    }
    else if(backend == BACKEND_CUDA)
    {
        return "cuda";
    }
    return "unknown";
}

bool initializeKernel(KernelResources& resources, int backend)
{
    resources.backend = backend;
    if(backend == BACKEND_SEQ || backend == BACKEND_OMP)
    {
        return true;
    }
    if(backend == BACKEND_CUDA)
    {
        fprintf(stderr, "This build has no CUDA backend yet");
        return false;
    }
    fprintf(stderr, "Unknown backend %d\n", backend);
    return false;
}

void releaseKernel(KernelResources& resources)
{
    resources.x.clear();
    resources.y.clear();
    resources.z.clear();
    resources.m.clear();
}

void computeForces(KernelResources& resources, const Vec4* pos, int n, int begin, int end,
                   Vec4* acc, const SimParams& params)
{
    if(resources.backend == BACKEND_SEQ)
    {
        forcesSequential(pos, n, begin, end, acc, params);
    }
    else if(resources.backend == BACKEND_OMP)
    {
        forcesOpenMP(resources, pos, n, begin, end, acc, params);
    }
}
