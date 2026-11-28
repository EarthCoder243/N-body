#pragma once
#include "nbody/types.h"
#include <memory>
#include <string>

namespace nbody {

enum class Backend { Seq, Omp, Cuda };

Backend parseBackend(const std::string& name);   // "seq" | "omp" | "cuda"
const char* backendName(Backend b);
bool backendAvailable(Backend b);

// Accelerations for particles [begin, end) against all n particles.
// acc is indexed globally: only acc[begin..end) is written.
// Self-interaction is not skipped (except in Seq): with eps2 > 0 it adds exactly zero.
class ForceKernel {
public:
    virtual ~ForceKernel() = default;
    virtual void compute(const Vec4* pos, std::size_t n, std::size_t begin, std::size_t end,
                         Vec4* acc, const Params& p) = 0;
};

std::unique_ptr<ForceKernel> makeKernel(Backend b);

// FLOPs counted per pairwise interaction for GFLOP/s (the usual convention).
constexpr double kFlopsPerInteraction = 20.0;

}  // namespace nbody