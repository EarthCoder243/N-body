#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nbody {

// Positions: (x, y, z, mass). Velocities and accelerations: (x, y, z, unused).
// 16 bytes, the same layout as CUDA float4 and the GUI's glm::vec4, so all positions
// move in a single MPI_Allgatherv.
struct alignas(16) Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
};
static_assert(sizeof(Vec4) == 16, "Vec4 must be 16 bytes");

struct Params {
    float dt = 0.0035f;   // same defaults as the GUI's -t and -s
    float eps2 = 40.0f;
    float G = 1.0f;
};

struct State {
    std::vector<Vec4> pos;   // all N particles, on every rank
    std::vector<Vec4> vel;   // indexed globally; a rank only advances its own slice
    std::vector<Vec4> acc;
    std::uint64_t step = 0;
    std::uint64_t seed = 0;
    Params params;

    std::size_t size() const { return pos.size(); }
    void resize(std::size_t n) { pos.resize(n); vel.resize(n); acc.resize(n); }
};

}  // namespace nbody