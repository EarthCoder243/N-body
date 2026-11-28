/*
Sequential reference kernel, same math as the GUI's. Bottom of the speedup ladder. 
Reference run to be compared against
*/

#include "forces.h"
#include <cmath>

void forcesSequential(const Vec4* pos, int n, int begin, int end, Vec4* acc, const SimParams& params)
{
    for(int i = begin; i < end; i++)
    {
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;
        for(int j = 0; j < n; j++)
        {
            if(j == i)
            {
                continue;
            }
            float dx = pos[j].x - pos[i].x;
            float dy = pos[j].y - pos[i].y;
            float dz = pos[j].z - pos[i].z;
            float dist3 = (float)std::pow(dx * dx + dy * dy + dz * dz + params.eps2, 1.5);
            float s = pos[j].w / dist3;
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }
        acc[i] = makeVec4(params.G * ax, params.G * ay, params.G * az, 0.0f);
    }
}
