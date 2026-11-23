#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include "ParticleSystemGalaxyInitializer.h"
#include "ParticleSolverCPUSequential.h"

int main(int argc, char *argv[])
{
    // Defaults (same time step and softening as the original ArgumentsParser)
    int version = 1;
    size_t numParticles = 100;
    int steps = 100;
    unsigned seed = 1;
    std::string outPath = "final.bin";
    float timeStep = .001f;
    float squaredSoftening = .0935f;

    // Parse -v -n -steps -seed -o
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "-v") version = std::stoi(argv[i + 1]);
        else if (arg == "-n") numParticles = std::stoul(argv[i + 1]);
        else if (arg == "-steps") steps = std::stoi(argv[i + 1]);
        else if (arg == "-seed") seed = std::stoul(argv[i + 1]);
        else if (arg == "-o") outPath = argv[i + 1];
    }

    if (version != 1) {
        std::cerr << "Only -v 1 is available in the headless build\n";
        return 1;
    }

    // Same world size as the original main.cpp
    glm::vec3 worldDimensions(5.f);

    // Build particles with a fixed seed so runs repeat exactly
    ParticleSystemGalaxyInitializer initializer(numParticles, seed);
    std::unique_ptr<ParticleSystem> particles = initializer.generateParticles(worldDimensions, false);

    ParticleSolverCPUSequential solver(timeStep, squaredSoftening);

    // Time each step and print step,ms rows (redirect stdout to a CSV file)
    std::cout << "step,ms\n";
    for (int step = 0; step < steps; step++) {
        auto start = std::chrono::steady_clock::now();
        solver.updateParticlePositions(particles.get());
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << step << "," << ms << "\n";
    }

    // Dump final positions as binary floats (x, y, z, w per particle)
    std::ofstream out(outPath, std::ios::binary);
    out.write(reinterpret_cast<const char *>(particles->getPositions()), sizeof(glm::vec4) * particles->size());

    return 0;
}