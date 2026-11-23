

#ifndef N_BODY_PARTICLESYSTEMGALAXYINITIALIZER_H
#define N_BODY_PARTICLESYSTEMGALAXYINITIALIZER_H
#include "ParticleSystemInitializer.h"
// CHANGED: needed for the clock-based default seed below
#include <chrono>


class ParticleSystemGalaxyInitializer: public ParticleSystemInitializer{
public:
    // CHANGED: optional seed; defaults to the clock, so the original main.cpp still works
    ParticleSystemGalaxyInitializer(size_t numParticles, unsigned seed = std::chrono::system_clock::now().time_since_epoch().count());
    std::unique_ptr<ParticleSystem> generateParticles(glm::vec3 worldDimensions, const bool usesGPU);

private:
    size_t totalParticles;
    // CHANGED: stored seed used by both random generators
    unsigned seed;

};


#endif //N_BODY_PARTICLESYSTEMGALAXYINITIALIZER_H