#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "ParticleSystemGalaxyInitializer.h"
#include "nbody/kernels.h"

//total energy (kinetic + softened potential) and total momentum, for the correctness check
void printEnergy(const nbody::State &state, const char *label)
{
    const std::size_t n = state.size();
    const nbody::Params &p = state.params;
    double kinetic = 0.0, potential = 0.0, px = 0.0, py = 0.0, pz = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        const nbody::Vec4 &v = state.vel[i];
        double m = state.pos[i].w;
        kinetic += 0.5 * m * (v.x * v.x + v.y * v.y + v.z * v.z);
        px += m * v.x; py += m * v.y; pz += m * v.z;
        for (std::size_t j = i + 1; j < n; j++) {
            double dx = state.pos[j].x - state.pos[i].x;
            double dy = state.pos[j].y - state.pos[i].y;
            double dz = state.pos[j].z - state.pos[i].z;
            potential -= p.G * m * state.pos[j].w / std::sqrt(dx * dx + dy * dy + dz * dz + p.eps2);
        }
    }
    std::cerr << label << ": energy " << kinetic + potential
              << " (kinetic " << kinetic << ", potential " << potential << ")"
              << ", momentum (" << px << ", " << py << ", " << pz << ")\n";
}

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

    //energy and momentum before the first step (printed to stderr so the CSV stays clean)
    printEnergy(state, "step 0");

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

    //energy and momentum after the last step
    printEnergy(state, ("step " + std::to_string(steps)).c_str());
    //dump final positions as (x, y, z, mass) floats per particle
    std::ofstream out(outPath, std::ios::binary);
    out.write(reinterpret_cast<const char *>(state.pos.data()), sizeof(nbody::Vec4) * numParticles);

    return 0;
}