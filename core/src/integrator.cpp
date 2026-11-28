/*
Leapfrog integration, energy diagnostic
*/
#include "integrator.h"
#include <cmath>
#include <omp.h>

void kick(SimState& state, int begin, int end, float h)
{
    #pragma omp parallel for schedule(static)
    for(int i = begin; i < end; i++)
    {
        state.vel[i].x += state.acc[i].x * h;
        state.vel[i].y += state.acc[i].y * h;
        state.vel[i].z += state.acc[i].z * h;
    }
}

void drift(SimState& state, int begin, int end, float h)
{
    #pragma omp parallel for schedule(static)
    for(int i = begin; i < end; i++)
    {
        state.pos[i].x += state.vel[i].x * h;
        state.pos[i].y += state.vel[i].y * h;
        state.pos[i].z += state.vel[i].z * h;
    }
}

void updateAccelerations(SimState& state, KernelResources& resources, int begin, int end)
{
    computeForces(resources, state.pos.data(), state.size(), begin, end, state.acc.data(), state.params);
}

StepTimes leapfrogStep(SimState& state, KernelResources& resources)
{
    int n = state.size();
    float half = 0.5f * state.params.dt;
    StepTimes times;

    double start = omp_get_wtime();
    kick(state, 0, n, half);
    drift(state, 0, n, state.params.dt);
    times.integrate = omp_get_wtime() - start;

    start = omp_get_wtime();
    updateAccelerations(state, resources, 0, n);
    times.force = omp_get_wtime() - start;

    start = omp_get_wtime();
    kick(state, 0, n, half);
    times.integrate += omp_get_wtime() - start;

    state.step++;
    return times;
}

Energy totalEnergy(const SimState& state)
{
    int n = state.size();
    double eps2 = state.params.eps2;
    double G = state.params.G;
    double kinetic = 0.0;
    double potential = 0.0;

    #pragma omp parallel for schedule(dynamic, 64) reduction (+:kinetic, potential)
    for(int i = 0; i < n; i++)
    {
        const Vec4& p = state.pos[i];
        const Vec4& v = state.vel[i];
        double mi = p.w;
        kinetic += 0.5 * mi * ((double)v.x * v.x + (double)v.y * v.y + (double)v.z * v.z);
        double phi = 0.0;
        for(int j = i + 1; j < n; j++)
        {
            double dx = (double)state.pos[j].x - p.x;
            double dy = (double)state.pos[j].y - p.y;
            double dz = (double)state.pos[j].z - p.z;
            phi -= state.pos[j].w / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
        potential += G * mi * phi;
    }

    Energy energy;
    energy.kinetic = kinetic;
    energy.potential = potential;
    return energy;
}
