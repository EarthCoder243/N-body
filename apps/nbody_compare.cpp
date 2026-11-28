/*
Correctness check between two state files or a run that crashed and resumed. 
Reports the max/RMS position difference relative to the system size. Should give 0.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "checkpoint.h"

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        fprintf(stderr, "Usage: nbody-compare A.bin B.bin [tolerance]\n");
        return 2;
    }
    double tolerance = (argc > 3) ? atof(argv[3]) : 1e-4;

    SimState a, b;
    if(!loadState(argv[1], a) || !loadState(argv[2], b))
    {
        return 2;
    }
    if(a.size() != b.size())
    {
        fprintf(stderr, "Particle counts differ: %d vs %d\n", a.size(), b.size());
        return 2;
    }
    if(a.step != b.step)
    {
        fprintf(stderr, "Warning: comparing step %llu with step %llu\n", (unsigned long long)a.step, (unsigned long long)b.step);
    }

    double maxPosError = 0.0;
    double sumSquares = 0.0;
    double maxVelError = 0.0;
    float low[3] = {1e30f, 1e30f, 1e30f}; //Bounding box for relative error for A
    float high[3] = {-1e30f, -1e30f, -1e30f};
    for(int i = 0; i < a.size(); i++)
    {
        const Vec4& p = a.pos[i];
        const Vec4& q = b.pos[i];
        double dx = (double)p.x - q.x;
        double dy = (double)p.y - q.y;
        double dz = (double)p.z - q.z;
        double d = std::sqrt(dx * dx + dy * dy + dz * dz);
        maxPosError = std::max(maxPosError, d);
        sumSquares += d * d;

        double vx = (double)a.vel[i].x - b.vel[i].x;
        double vy = (double)a.vel[i].y - b.vel[i].y;
        double vz = (double)a.vel[i].z - b.vel[i].z;
        maxVelError = std::max(maxVelError, std::sqrt(vx * vx + vy * vy + vz * vz));

        float coords[3] = {p.x, p.y, p.z};
        for(int k = 0; k < 3; k++)
        {
            low[k] = std::min(low[k], coords[k]);
            high[k] = std::max(high[k], coords[k]);
        }
    }

    double extent = std::max(high[0] - low[0], std::max(high[1] - low[1], high[2] - low[2]));
    double relError = (extent > 0.0) ? maxPosError / extent : maxPosError;
    bool match = relError <= tolerance;
    printf("n=%d,max_pos_err=%.3e,rms_pos_err=%.3e,rel_max_pos_err=%.3e,max_vel_err=%.3e,tol=%.1e->%s\n",
            a.size(), maxPosError, std::sqrt(sumSquares / a.size()), relError, maxVelError, tolerance, match ? "MATCH" : "DIFFER");
    return match ? 0 : 1;
}
