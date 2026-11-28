/*
Particle data shared by every part of the simulation.
Each rank keeps all N positions, but only moves its own slice,
gets rest of positions from MPI_Allgatherv
*/

#ifndef NBODY_PARTICLES_H
#define NBODY_PARTICLES_H

#include <vector> //particle arrays
#include <cstdint> //Fixed size step

const float WORLD_SIZE = 5.0f; //same world size as GUI
const float DEFAULT_DT = 0.0035f;
const float DEFAULT_EPS2 = 40.0f;
const double FLOPS_PER_INTERACTION = 20.0;

struct alignas(16) Vec4
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

struct SimParams
{
    float dt = DEFAULT_DT;
    float eps2 = DEFAULT_EPS2;
    float G = 1.0f;
};

struct SimState
{
    std::vector<Vec4> pos;
    std::vector<Vec4> vel;
    std::vector<Vec4> acc;
    uint64_t step = 0;
    uint64_t seed = 0;
    SimParams params;

    int size() const
    {
        return (int)pos.size();
    }

    void resize(int n)
    {
        pos.assign(n, Vec4());
        vel.assign(n, Vec4());
        acc.assign(n, Vec4());
    }
};

inline Vec4 makeVec4(float x, float y, float z, float w)
{
    Vec4 v;
    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;
    return v;
}

#endif
