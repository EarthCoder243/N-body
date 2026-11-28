/*
Initial particle setups. Same seed gives same particles on every rank, so MPI ranks build the starting state themselves instead of recieving
*/

#ifndef NBODY_INITIAL_CONDITIONS_H
#define NBODY_INITIAL_CONDITIONS_H

#include <string>
#include "particles.h"

const int INIT_DISK = 0;
const int INIT_CUBE = 1;
const int INIT_BALL = 2;

int parseInit(const std::string& name); //"disk", "cube", "ball", returns -1 if unknown
void createInitialState(SimState& state, int initType, int n, uint64_t seed, const SimParams& params);

#endif
