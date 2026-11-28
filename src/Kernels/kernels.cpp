#include "nbody/kernels.h"
#include <cmath>
#include <stdexcept>

namespace nbody {

// Reference kernel: the original ParticleSolverCPUSequential force loop, rewritten for Vec4
class SeqKernel : public ForceKernel {
public:
    void compute(const Vec4* pos, std::size_t n, std::size_t begin, std::size_t end,
                 Vec4* acc, const Params& p) override {
        for (std::size_t i = begin; i < end; i++) {
            float ax = 0.f, ay = 0.f, az = 0.f;
            for (std::size_t j = 0; j < n; j++) {
                if (i != j) {
                    // Only x, y, z: w holds the mass, not part of the distance
                    float dx = pos[j].x - pos[i].x;
                    float dy = pos[j].y - pos[i].y;
                    float dz = pos[j].z - pos[i].z;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    float dist = std::pow(r2 + p.eps2, 1.5f);
                    float s = (p.G * pos[j].w) / dist;
                    ax += s * dx;
                    ay += s * dy;
                    az += s * dz;
                }
            }
            acc[i] = {ax, ay, az, 0.f};
        }
    }
};

Backend parseBackend(const std::string& name) {
    if (name == "seq") return Backend::Seq;
    if (name == "omp") return Backend::Omp;
    if (name == "cuda") return Backend::Cuda;
    throw std::invalid_argument("unknown backend: " + name);
}

const char* backendName(Backend b) {
    switch (b) {
        case Backend::Seq: return "seq";
        case Backend::Omp: return "omp";
        case Backend::Cuda: return "cuda";
    }
    return "unknown";
}

// Only Seq exists for now; Omp and Cuda are added in later tasks
bool backendAvailable(Backend b) {
    return b == Backend::Seq;
}

std::unique_ptr<ForceKernel> makeKernel(Backend b) {
    if (b == Backend::Seq) return std::make_unique<SeqKernel>();
    return nullptr;
}

}  // namespace nbody