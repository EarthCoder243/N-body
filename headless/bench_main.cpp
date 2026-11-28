#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "ParticleSystemGalaxyInitializer.h"
#include "nbody/kernels.h"

int main(int argc, char *argv[])
{
    //time step and softening come from nbody::Params (shared with the MPI code)
    std::string backendArg = "seq";
    size_t numParticles = 100;
    int steps = 100;
    unsigned seed = 1;
    std::string outPath = "final.bin";
    nbody::Params params;

    //-backend replaces -v; -t and -s override the Params defaults
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "-backend") backendArg = argv[i + 1];
        else if (arg == "-n") numParticles = std::stoul(argv[i + 1]);
        else if (arg == "-steps") steps = std::stoi(argv[i + 1]);
        else if (arg == "-seed") seed = std::stoul(argv[i + 1]);
        else if (arg == "-o") outPath = argv[i + 1];
        else if (arg == "-t") params.dt = std::stof(argv[i + 1]);
        else if (arg == "-s") params.eps2 = std::stof(argv[i + 1]);
    }

    nbody::Backend backend = nbody::parseBackend(backendArg);
    if (!nbody::backendAvailable(backend)) {
        std::cerr << "Backend " << nbody::backendName(backend) << " is not available in this build\n";
        return 1;
    }
    std::unique_ptr<nbody::ForceKernel> kernel = nbody::makeKernel(backend);

    // Same world size as the original main.cpp
    glm::vec3 worldDimensions(5.f);

    // Build particles with a fixed seed so runs repeat exactly
    ParticleSystemGalaxyInitializer initializer(numParticles, seed);
    std::unique_ptr<ParticleSystem> particles = initializer.generateParticles(worldDimensions, false);

    //copy the particles into nbody::State (mass goes in pos.w)
    nbody::State state;
    state.resize(numParticles);
    state.params = params;
    state.seed = seed;
    for (size_t i = 0; i < numParticles; i++) {
        glm::vec4 p = particles->getPositions()[i];
        glm::vec4 v = particles->getVelocities()[i];
        glm::vec4 a = particles->getAccelerations()[i];
        state.pos[i] = {p.x, p.y, p.z, particles->getMasses()[i].x};
        state.vel[i] = {v.x, v.y, v.z, 0.f};
        state.acc[i] = {a.x, a.y, a.z, 0.f};
    }

    //new accelerations from the kernel, kept apart from state.acc until the update
    std::vector<nbody::Vec4> newAcc(numParticles);
    float half = params.dt * 0.5f;

    // Time each step and print step,ms rows (redirect stdout to a CSV file)
    std::cout << "step,ms\n";
    for (int step = 0; step < steps; step++) {
        auto start = std::chrono::steady_clock::now();

        //forces from the kernel for all particles
        kernel->compute(state.pos.data(), numParticles, 0, numParticles, newAcc.data(), params);

        // CHANGED: same leapfrog as ParticleSystem::updateParticlePosition
        for (size_t i = 0; i < numParticles; i++) {
            nbody::Vec4 &x = state.pos[i];
            nbody::Vec4 &v = state.vel[i];
            nbody::Vec4 &a = state.acc[i];
            v.x += a.x * half; v.y += a.y * half; v.z += a.z * half;
            x.x += v.x * params.dt; x.y += v.y * params.dt; x.z += v.z * params.dt;
            a = newAcc[i];
            v.x += a.x * half; v.y += a.y * half; v.z += a.z * half;
        }
        state.step++;

        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << step << "," << ms << "\n";
    }

    //dump final positions as (x, y, z, mass) floats per particle
    std::ofstream out(outPath, std::ios::binary);
    out.write(reinterpret_cast<const char *>(state.pos.data()), sizeof(nbody::Vec4) * numParticles);

    return 0;
}