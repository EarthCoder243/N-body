/*
OpenMP + SIMD kernel. Positions are copies into seperate x, y, z mass arrays each call
*/

#include "forces.h"
#include <cmath>
#include <omp.h>

void forcesOpenMP(KernelResources& resources, const Vec4* pos, int n, int begin, int end,
                   Vec4* acc, const SimParams& params)
{
    resources.x.resize(n);
    resources.y.resize(n);
    resources.z.resize(n);
    resources.m.resize(n);
    float* xs = resources.x.data();
    float* ys = resources.y.data();
    float* zs = resources.z.data();
    float* ms = resources.m.data();
    float eps2 = params.eps2;
    float G = params.G;

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for(int j = 0; j < n; j++)
        {
            xs[j] = pos[j].x;
            ys[j] = pos[j].y;
            zs[j] = pos[j].z;
            ms[j] = pos[j].w;
        }

        #pragma omp for schedule(static)
        for(int i = begin; i < end; i++)
        {
            float xi = xs[i];
            float yi = ys[i];
            float zi = zs[i];
            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;
            #pragma omp simd reduction(+:ax, ay, az)
            for(int j = 0; j < n; j++)
            {
                float dx = xs[j] - xi;
                float dy = ys[j] - yi;
                float dz = zs[j] - zi;
                float r2 = dx * dx + dy * dy + dz * dz + eps2;
                float inv = 1.0f / std::sqrt(r2);
                float s = ms[j] * inv * inv * inv;
                ax += dx * s;
                ay += dy * s;
                az += dz * s;
            }
            acc[i] = makeVec4(G * ax, G * ay, G * az, 0.0f);
        }
    }
}
