/*
Kick-drift-kick leapfrog. Unlike GUI ParticleSystem::updateParticlePosition, second half kick uses forces at the new positions
*/

#ifndef NBODY_INTEGRATOR_H
#define NBODY_INTEGRATOR_H

#include "particles.h"
#include "forces.h"

struct StepTimes
{
    double force = 0.0;
    double integrate = 0.0;
};

struct Energy
{
    double kinetic = 0.0;
    double potential = 0.0;

    double total() const
    {
        return kinetic + potential;
    }
};

void kick(SimState& state, int begin, int end, float h);
void drift(SimState& state, int begin, int end, float h);
void updateAccelerations(SimState& state, KernelResources& resources, int begin, int end);
StepTimes leapfrogStep(SimState& state, KernelResources& resources);
Energy totalEnergy(const SimState& state);

#endif
