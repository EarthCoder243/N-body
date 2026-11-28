/*
Distributed N-body with MPI. Each rank owns slice of particles, sized by measured throughput, every step shares updated positions with all ranks.
Every K steps, all ranks also share velocities, one rnk writes checkpoint

--results appends one CSV row per run.
*/

#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "forces.h"
#include "integrator.h"
#include "initial_conditions.h"
#include "checkpoint.h"

struct Options //CLI settings, default match GUI when possible
{
    int backend = BACKEND_OMP;
    int initType = INIT_DISK;
    int n = 2048;
    uint64_t steps = 100;
    uint64_t seed = 23;
    SimParams params;
    int threads = 0; //0 OpenMP default
    bool equalWeights = false;
    std::string outFile;
    std::string resultsFile;
    std::string label = "run";
    std::string rankLog;
    std::string ckptDir;
    uint64_t ckptEvery = 0; //0 = no checkpoints
    bool resume = false;
    bool help = false;
};

struct StateHeader
{
    uint64_t n;
    uint64_t step;
    uint64_t seed;
    SimParams params;
};

/*
Prints the error with rank number and kills every rank
*/
void abortRun(int rank, const std::string& msg)
{
    fprintf(stderr, "[rank %d] %s\n", rank, msg.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
}

void printUsage()
{
    fprintf(stderr,
            "usage: mpirun -np P [--bind-to none] nbody-mpi [options]\n"
            "   --backend seq|omp|cuda              force kernel (default omp)\n"
            "   --init disk|ball|cube               initial conditions (default disk)\n"
            "   -n N                               particles (default 2048)\n"
            "   --steps S                           run until step S (default 100)\n"
            "   --seed X                            random seed (default 23)\n"
            "   --dt T --eps2 E                     time step and squared softening (default 0.0035, 40)\n"
            "   --threads T                         OpenMP threads\n"
            "   --weights measured|equal            slice sizes from calibration or equal (default measured)\n"
            "   --out FILE                          write the final state\n"
            "   --results FILE --label NAME         append a CSV row for this run\n"
            "   --ckpt-dir DIR --ckpt-every K       checkpoint every K steps\n"
            "   --resume                            continue from the newest checkpoint in --ckpt-dir\n"
            "   --rank-log PREFIX                   per rank step timings to PREFIX-rank<r>.csv\n");
}

/*
Reads CLI flags into opt. Every rank parses same args, only rank 0 prints errors
*/

bool parseOptions(int argc, char** argv, Options& opt, int rank)
{
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "-h" || arg == "--help" || arg == "--h")
        {
            opt.help = true;
            return false;
        }
        else if(arg == "--resume")
        {
            opt.resume = true;
        }
        else if(arg[0] == '-' && i + 1 < argc)
        {
            std::string value = argv[++i];
            if(arg == "--backend")
            {
                opt.backend = parseBackend(value);
            }
            else if(arg == "--init")
            {
                opt.initType = parseInit(value);
            }
            else if(arg == "-n")
            {
                opt.n = atoi(value.c_str());
            }
            else if(arg == "--steps")
            {
                opt.steps = strtoull(value.c_str(), NULL, 10);
            }
            else if(arg == "--seed")
            {
                opt.seed = strtoull(value.c_str(), NULL, 10);
            }
            else if(arg == "--dt")
            {
                opt.params.dt = (float)atof(value.c_str());
            }
            else if(arg =="--eps2")
            {
                opt.params.eps2 = (float)atof(value.c_str());
            }
            else if(arg == "--threads")
            {
                opt.threads = atoi(value.c_str());
            }
            else if(arg == "--weights")
            {
                opt.equalWeights = (value == "equal");
            }
            else if(arg == "--out")
            {
                opt.outFile = value;
            }
            else if(arg == "--results")
            {
                opt.resultsFile = value;
            }
            else if(arg == "--label")
            {
                opt.label = value;
            }
            else if(arg == "--ckpt-dir")
            {
                opt.ckptDir = value;
            }
            else if(arg == "--ckpt-every")
            {
                opt.ckptEvery = strtoull(value.c_str(), NULL, 10);
            }
            else if(arg == "--rank-log")
            {
                opt.rankLog = value;
            }
            else
            {
                if(rank == 0)
                {
                    fprintf(stderr, "Unknown option %s\n", arg.c_str());
                }
                return false;
            }
        }
        else
        {
            if(rank == 0)
            {
                fprintf(stderr, "Unknown or incomplete option %s\n", arg.c_str());
            }
            return false;
        }
    }
    std::string error;
    if(opt.backend < 0)
    {
        error = "Backend must be seq, omp, or cuda";
    }
    else if(opt.initType < 0)
    {
        error = "Init must be disk, cube, or ball";
    }
    else if(opt.n <= 0 || opt.params.dt <= 0.0f || opt.params.eps2 <= 0.0f)
    {
        error = "-n, --dt, and --eps2 must be positive";
    }
    else if((opt.resume || opt.ckptEvery > 0) && opt.ckptDir.empty())
    {
        error = "--resume and --ckpt-every need --ckpt-dir";
    }
    if(!error.empty())
    {
        if(rank == 0)
        {
            fprintf(stderr, "%s\n", error.c_str());
        }
        return false;
    }
    return true;
}

/*
Times this rank's kernel on a sample of particles against all n.
Every rank runs on the same sample, so faster machines get proportionally bigger slices
*/
double measureThroughput(SimState& state, KernelResources& resources)
{
    int n = state.size();
    int sample = (n < 8192) ? n : 8192;
    updateAccelerations(state, resources, 0, sample);
    double best = 0.0;

    for(int r = 0; r < 3; r++)
    {
        double start = omp_get_wtime();
        updateAccelerations(state, resources, 0, sample);
        double elapsed = omp_get_wtime() - start;
        if(best == 0.0 || elapsed < best)
        {
            best = elapsed;
        }
    }
    return (best > 0.0) ? (double)sample * (double)n / best : 1.0;
}

/*
Splits n particles into contiguous slices proportional to weights, computed by each rank from the gathered weights
*/
void computeSlices(int n, const std::vector<double>& weights, std::vector<int>& counts, std::vector<int>& displs)
{
    int size = (int)weights.size();
    double total = 0.0;
    for(int r = 0; r < size; r++)
    {
        total += weights[r];
    }
    int assigned = 0;
    for(int r = 0; r < size; r++)
    {
        counts[r] = (int)std::floor((double)n * weights[r] / total);
        assigned += counts[r];
    }
    for(int r = 0; assigned < n; r = (r + 1) % size)
    {
        counts[r]++;
        assigned++;
    }
    displs[0] = 0;
    for(int r = 1; r < size; r++)
    {
        displs[r] = displs[r - 1] + counts[r - 1];
    }
}

/*
Every rank sends its slice of velocities to every other rank, needed before writing checkpoint and before re-slicing in load balancer
*/
void gatherVelocities(SimState& state, std::vector<int>& counts, std::vector<int>& displs, MPI_Datatype vec4Type)
{
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.vel.data(), counts.data(), displs.data(), vec4Type, MPI_COMM_WORLD);
}

/*
Finds the newest step that every rank with checkpoints has, loads it on lowest rank that has it, and broadcasts to everyone. Returns false if no rank has a checkpoint
*/
bool resumeFromCheckpoint(SimState& state, const std::string& dir, int rank, int size, bool checkpointWriter, MPI_Datatype vec4Type)
{
    uint64_t newest = 0;
    uint64_t mine = newestCheckpoint(dir, newest) ? newest : UINT64_MAX;
    uint64_t common;
    MPI_Allreduce(&mine, &common, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    if(common == UINT64_MAX)
    {
        if(rank == 0)
        {
            printf("No checkpoints on any rank, starting new\n");
        }
        return false;
    }
    int candidate = hasCheckpoint(dir, common) ? rank : size;
    int source;
    MPI_Allreduce(&candidate, &source, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    StateHeader header = {};
    if(rank == source)
    {
        if(!loadState(checkpointPath(dir, common), state))
        {
            abortRun(rank, "Unable to load checkpoint " + checkpointPath(dir, common));
        }
        header.n = (uint64_t)state.size();
        header.step = state.step;
        header.seed = state.seed;
        header.params = state.params;
    }
    MPI_Bcast(&header, sizeof(StateHeader), MPI_BYTE, source, MPI_COMM_WORLD);
    if(rank != source)
    {
        state.resize((int)header.n);
        state.step = header.step;
        state.seed = header.seed;
        state.params = header.params;
    }
    MPI_Bcast(state.pos.data(), (int)header.n, vec4Type, source, MPI_COMM_WORLD);
    MPI_Bcast(state.vel.data(), (int)header.n, vec4Type, source, MPI_COMM_WORLD);

    if(checkpointWriter)
    {
        discardCheckpointsAfter(dir, common); //Clears steps newer than the resume point from failed run
    }
    if(rank == 0)
    {
        printf("Resumed at step %llu from rank %d's checkpoint\n", (unsigned long long)common, source);
    }
    return true;
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Options opt;
    if(!parseOptions(argc, argv, opt, rank))
    {
        if(rank == 0)
        {
            printUsage();
        }
        MPI_Finalize();
        return opt.help ? 0 : 2;
    }
    if(opt.threads > 0)
    {
        omp_set_num_threads(opt.threads);
    }
    int threads = (opt.backend == BACKEND_SEQ) ? 1 : omp_get_max_threads();

    MPI_Comm nodeComm; //Ranks on the same machine share one checkpoint directory
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank;
    MPI_Comm_rank(nodeComm, &nodeRank);
    bool checkpointWriter = (nodeRank == 0); //One writer per node

    MPI_Datatype vec4Type; //4 floats, matches Vec4
    MPI_Type_contiguous(4, MPI_FLOAT, &vec4Type);
    MPI_Type_commit(&vec4Type);

    char host[MPI_MAX_PROCESSOR_NAME];
    int hostLen;
    MPI_Get_processor_name(host, &hostLen);

    KernelResources resources;
    if(!initializeKernel(resources, opt.backend))
    {
        abortRun(rank, std::string("Unable to start backend ") + backendName(opt.backend));
    }

    SimState state;
    bool resumed = false;
    if(opt.resume)
    {
        resumed = resumeFromCheckpoint(state, opt.ckptDir, rank, size, checkpointWriter, vec4Type);
    }
    if(!resumed)
    {
        createInitialState(state, opt.initType, opt.n, opt.seed, opt.params);
    }
    int n = state.size();

    //Calibration: slice sizes proportional to throughput
    std::vector<double> weights(size, 1.0);
    double throughput = 0.0;
    if(!opt.equalWeights)
    {
        throughput = measureThroughput(state, resources);
        MPI_Allgather(&throughput, 1, MPI_DOUBLE, weights.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    std::vector<int> counts(size), displs(size);
    computeSlices(n, weights, counts, displs);
    int begin = displs[rank];
    int end = begin + counts[rank];

    printf("[rank %d] host=%s,backend=%s,threads=%d,throughput=%.3e,begin=%d,count=%d\n",
           rank, host, backendName(opt.backend), threads, throughput, begin, counts[rank]);
    fflush(stdout);

    FILE* rankLog = NULL;
    if(!opt.rankLog.empty())
    {
        std::string path = opt.rankLog + "-rank" + std::to_string(rank) + ".csv";
        rankLog = fopen(path.c_str(), "w");
        if(rankLog == NULL)
        {
            abortRun(rank, "Unable to open " + path);
        }
        fprintf(rankLog, "step,compute_s,comm_s\n");
    }

    updateAccelerations(state, resources, begin, end);

    float half = 0.5f * state.params.dt;
    double computeTime = 0.0;
    double commTime = 0.0;
    double ckptTime = 0.0;
    uint64_t timedSteps = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double loopStart = MPI_Wtime();
    while(state.step < opt.steps) //Same sequence as leapfrog step on this rank's slice
    {
        double t0 = MPI_Wtime();
        kick(state, begin, end, half);
        drift(state, begin, end, state.params.dt);
        double t1 = MPI_Wtime();
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.pos.data(), counts.data(), displs.data(), vec4Type, MPI_COMM_WORLD); //every rank gets new position
        double t2 = MPI_Wtime();
        updateAccelerations(state, resources, begin, end);
        kick(state, begin, end, half);
        state.step++;
        double t3 = MPI_Wtime();

        double stepCompute = (t1 - t0) + (t3 - t2);
        double stepComm = t2 - t1; //Includes waiting for slower ranks
        computeTime += stepCompute;
        commTime += stepComm;
        timedSteps++;
        if(rankLog)
        {
            fprintf(rankLog, "%llu,%.6e,%.6e\n", (unsigned long long)state.step, stepCompute, stepComm);
        }
        if(opt.ckptEvery > 0 && state.step % opt.ckptEvery == 0)
        {
            double c0 = MPI_Wtime();
            gatherVelocities(state, counts, displs, vec4Type); //Every rank has full state
            if(checkpointWriter && !writeCheckpoint(opt.ckptDir, state))
            {
                abortRun(rank, "Unable to write checkpoint in " + opt.ckptDir);
            }
            ckptTime += MPI_Wtime() - c0;
        }
    }
    double wallTime = MPI_Wtime() - loopStart;
    if(rankLog)
    {
        fclose(rankLog);
    }

    if(!opt.outFile.empty())
    {
        gatherVelocities(state, counts, displs, vec4Type);
        if(rank == 0 && !saveState(opt.outFile, state))
        {
            abortRun(rank, "Unable to write " + opt.outFile);
        }
    }

    //Run summary on rank 0, slowest rank sets the pace, imbalance = slowest compute / mean compute
    double local[4] = {wallTime, computeTime, commTime, ckptTime};
    double maxStats[4], sumStats[4];
    MPI_Reduce(local, maxStats, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, sumStats, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        double steps = (timedSteps > 0) ? (double)timedSteps : 1.0;
        double stepMs = 1000.0 * maxStats[0] / steps;
        double computeMs = 1000.0 * maxStats[1] / steps;
        double commMs = 1000.0 * maxStats[2] / steps;
        double ckptMs = 1000.0 * maxStats[3];
        double meanCompute = sumStats[1] / size;
        double imbalance = (meanCompute > 0.0) ? maxStats[1] / meanCompute : 1.0;
        double interactionsPerSecond = (maxStats[0] > 0.0) ? (double) n * (double)n * (double)timedSteps / maxStats[0] : 0.0;
        double gflops = interactionsPerSecond * FLOPS_PER_INTERACTION * 1e-9;

        printf("label=%s,ranks=%d,backend=%s,n=%d,steps=%llu,final_step=%llu,threads=%d,step_ms=%.3f,compute_ms=%.3f,"
              "comm_ms=%.3f,ckpt_ms=%.3f,imbalance=%.3f,interactions_per_second=%.4e,gflops=%.2f\n",
              opt.label.c_str(), size, backendName(opt.backend), n, (unsigned long long)timedSteps,
              (unsigned long long)state.step, threads, stepMs, computeMs, commMs, ckptMs, imbalance,
              interactionsPerSecond, gflops);
        
        if(!opt.resultsFile.empty())
        {
            FILE* f = fopen(opt.resultsFile.c_str(), "a");
            if(f)
            {
                fprintf(f, "%s,%d,%s,%d,%llu,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.4e,%.2f\n",
                        opt.label.c_str(), size, backendName(opt.backend), n, (unsigned long long)timedSteps,
                        threads, stepMs, computeMs, commMs, ckptMs, imbalance, interactionsPerSecond, gflops);
                fclose(f);
            }
            else
            {
                fprintf(stderr, "Unable to open %s\n", opt.resultsFile.c_str());
            }
        }
    }

    releaseKernel(resources);
    MPI_Type_free(&vec4Type);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return 0;
}

