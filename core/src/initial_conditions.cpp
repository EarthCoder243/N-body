/*
Disk (adapted from GUI's galaxy initializer), cube, and ball setups. Masses scale with 1/N so total mass is the same for any N
*/

#include "initial_conditions.h"
#include <cmath>
#include <random>

const float PI = 3.14159265f;

int parseInit(const std::string& name)
{
    if(name == "disk")
    {
        return INIT_DISK;
    }
    else if(name == "cube")
    {
        return INIT_CUBE;
    }
    else if(name == "ball")
    {
        return INIT_BALL;
    }
    return -1;
}

void createInitialState(SimState& state, int initType, int n, uint64_t seed, const SimParams& params)
{
    state.resize(n);
    state.params = params;
    state.seed = seed;
    state.step = 0;

    std::mt19937 gen((unsigned int)seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    float center = WORLD_SIZE / 2.0f;
    float baseMass = WORLD_SIZE * WORLD_SIZE * WORLD_SIZE / (float)n;

    if(initType == INIT_DISK)
    {
        float speed = std::sqrt(3.0f * WORLD_SIZE * WORLD_SIZE) / 4.0f;
        float bulgeRadius = WORLD_SIZE / 10.0f;
        for(int i = 0; i < n; i++)
        {
            float radius = WORLD_SIZE * unit(gen);
            float theta = 2.0f * PI * unit(gen);
            float height = (WORLD_SIZE / 10.0f) * (2.0f * unit(gen) - 1.0f);
            float mass = baseMass * ((radius < bulgeRadius) ? 0.7f : 0.5f);
            state.pos[i] = makeVec4(center + radius * std::cos(theta), center + radius * std::sin(theta), center + height, mass);
            state.vel[i] = makeVec4(-speed * std::sin(theta), speed * std::cos(theta), 0.0f, 0.0f);
        }
    }
    else if(initType == INIT_CUBE)
    {
        for(int i = 0; i < n; i++)
        {
            float x = WORLD_SIZE * unit(gen);
            float y = WORLD_SIZE * unit(gen);
            float z = WORLD_SIZE * unit(gen);
            state.pos[i] = makeVec4(x, y, z, baseMass * 0.5f);
        }
    }
    else
    {
        for(int i = 0; i < n; i++)
        {
            float x, y, z;
            do
            {
                x = 2.0f * unit(gen) - 1.0f;
                y = 2.0f * unit(gen) - 1.0f;
                z = 2.0f * unit(gen) - 1.0f;
            } while(x * x + y * y + z * z > 1.0f);
            state.pos[i] = makeVec4(center + center * x, center + center * y, center + center * z, baseMass * 0.5f);
        }
    }
}
