#include "nbody/kernels.h"
#include <cmath>
#include <stdexcept>
#include <vector>

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

// Optimized CPU kernel: SoA copies, 1/sqrt instead of pow, no i == j branch, OpenMP + SIMD
class OmpKernel : public ForceKernel {
public:
    void compute(const Vec4* pos, std::size_t n, std::size_t begin, std::size_t end,
                 Vec4* acc, const Params& p) override {
        // SoA: separate x, y, z, m arrays so the inner loop reads contiguous floats
        x.resize(n); y.resize(n); z.resize(n); m.resize(n);
        for (std::size_t j = 0; j < n; j++) {
            x[j] = pos[j].x; y[j] = pos[j].y; z[j] = pos[j].z; m[j] = p.G * pos[j].w;
        }
        const float* xs = x.data(); const float* ys = y.data();
        const float* zs = z.data(); const float* ms = m.data();
        const float eps2 = p.eps2;

        #pragma omp parallel for schedule(static)
        for (std::size_t i = begin; i < end; i++) {
            const float xi = xs[i], yi = ys[i], zi = zs[i];
            float ax = 0.f, ay = 0.f, az = 0.f;
            // No i == j check: with eps2 > 0 the self term has dx = dy = dz = 0, so it adds 0
            #pragma omp simd reduction(+:ax, ay, az)
            for (std::size_t j = 0; j < n; j++) {
                float dx = xs[j] - xi;
                float dy = ys[j] - yi;
                float dz = zs[j] - zi;
                float r2 = dx * dx + dy * dy + dz * dz + eps2;
                float inv = 1.0f / std::sqrt(r2);
                float s = ms[j] * inv * inv * inv;
                ax += s * dx;
                ay += s * dy;
                az += s * dz;
            }
            acc[i] = {ax, ay, az, 0.f};
        }
    }

private:
    std::vector<float> x, y, z, m;
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
    return b == Backend::Seq || b == Backend::Omp;
}

std::unique_ptr<ForceKernel> makeKernel(Backend b) {
    if (b == Backend::Seq) return std::make_unique<SeqKernel>();
    if (b == Backend::Omp) return std::make_unique<OmpKernel>();
    return nullptr;
}

}  // namespace nbody