/*
Checkpoint files. Accelerations aren't stored, they're recomputed from positions after loading. 
Fault tolerance: every node keeps its own copy. On resume, ranks agree on newest step they all have with MPI_Allreduce(MIN)
*/

#ifndef NBODY_CHECKPOINT_H
#define NBODY_CHECKPOINT_H

#include <string>
#include "particles.h"

bool saveState(const std::string& path, const SimState& state);
bool loadState(const std::string& path, SimState& state);
std::string checkpointPath(const std::string& dir, uint64_t step);

bool writeCheckpoint(const std::string& dir, const SimState& state, int keep = 2);
bool newestCheckpoint(const std::string& dir, uint64_t& step);
bool hasCheckpoint(const std::string& dir, uint64_t step);
void discardCheckpointsAfter(const std::string& dir, uint64_t step);

#endif
