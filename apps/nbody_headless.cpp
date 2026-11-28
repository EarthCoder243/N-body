/*
Single process benchmark and reference run. Used as a reference

--results appends one CSV row per run, no header
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <omp.h>
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
    int warmup = 2; //Untimed force calculations before timing starts
    std::string outFile;
    std::string resultsFile;
    std::string label = "run";
    std::string ckptDir;
    uint64_t ckptEvery = 0; //0 = no checkpoints
    bool resume = false;
    bool energy = false;
};

void printUsage()
{
    fprintf(stderr,
            "usage: nbody-headless [options]\n"
            "   --backend seq|omp|cuda              force kernel (default omp)\n"
            "   --init disk|ball|cube               initial conditions (default disk)\n"
            "   -n N                               particles (default 2048)\n"
            "   --steps S                           run until step S (default 100)\n"
            "   --seed X                            random seed (default 23)\n"
            "   --dt T --eps2 E                     time step and squared softening (default 0.0035, 40)\n"
            "   --threads T                         OpenMP threads\n"
            "   --warmup W                          untimed force calculations first (default 2)\n"
            "   --out FILE                          write the final state\n"
            "   --results FILE --label NAME         append a CSV row for this run\n"
            "   --ckpt-dir DIR --ckpt-every K       checkpoint every K steps\n"
            "   --resume                            continue from the newest checkpoint in --ckpt-dir\n"
            "   --energy                            report energy drift (costs about one extra step)\n");
}

bool parseOptions(int argc, char** argv, Options& opt)
{
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "-h" || arg == "--help" || arg == "--h")
        {
            printUsage();
            exit(0);
        }
        else if(arg == "--resume")
        {
            opt.resume = true;
        }
        else if(arg == "--energy")
        {
            opt.energy = true;
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
            else if(arg == "--warmup")
            {
                opt.warmup = atoi(value.c_str());
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
            else
            {
                fprintf(stderr, "Unknown option %s\n", arg.c_str());
                return false;
            }
        }
        else
        {
            fprintf(stderr, "Unknown option or incomplete option %s\n", arg.c_str());
            return false;
        }
    }

    if(opt.backend < 0)
    {
        fprintf(stderr, "Backend must be seq, omp, or cuda\n");
        return false;
    }
    if(opt.initType < 0)
    {
        fprintf(stderr, "Init must be disk, cube, or ball\n");
        return false;
    }
    if(opt.n <= 0 || opt.params.dt <= 0.0f || opt.params.eps2 <= 0.0f)
    {
        fprintf(stderr, "-n, --dt, and --eps2 must be positive\n");
        return false;
    }
    if((opt.resume || opt.ckptEvery > 0) && opt.ckptDir.empty())
    {
        fprintf(stderr, "--resume and --ckpt-every need --ckpt-dir\n");
        return false;
    }
    return true;
}

int main(int argc, char** argv)
{
    Options opt;
    if(!parseOptions(argc, argv, opt))
    {
        printUsage();
        return 2;
    }
    if(opt.threads > 0)
    {
        omp_set_num_threads(opt.threads);
    }
    int threads = omp_get_max_threads();
    if(opt.backend == BACKEND_SEQ)
    {
        threads = 1;
    }

    KernelResources resources;
    if(!initializeKernel(resources, opt.backend))
    {
        return 1;
    }
    SimState state;
    if(opt.resume)
    {
        uint64_t step;
        if(!newestCheckpoint(opt.ckptDir, step))
        {
            fprintf(stderr, "No complete checkpoint in %s\n", opt.ckptDir.c_str());
            return 1;
        }
        if(!loadState(checkpointPath(opt.ckptDir, step), state))
        {
            return 1;
        }
        discardCheckpointsAfter(opt.ckptDir, step);
        fprintf(stderr, "Resumed at step %llu (n, seed, dt, and eps2 come from the checkpoint)\n",
                (unsigned long long)step);
    }
    else
    {
        createInitialState(state, opt.initType, opt.n, opt.seed, opt.params);
    }
    int n = state.size();

    for(int w = 0; w <= opt.warmup; w++)
    {
        updateAccelerations(state, resources, 0, n);
    }

    Energy startEnergy;
    if(opt.energy)
    {
        startEnergy = totalEnergy(state);
    }

    double forceTime = 0.0;
    double integrateTime = 0.0;
    uint64_t timedSteps = 0;
    while(state.step < opt.steps)
    {
        StepTimes times = leapfrogStep(state, resources);
        forceTime += times.force;
        integrateTime += times.integrate;
        timedSteps++;
        if(opt.ckptEvery > 0 && state.step % opt.ckptEvery == 0)
        {
            if(!writeCheckpoint(opt.ckptDir, state))
            {
                return 1;
            }
        }
    }
    if(!opt.outFile.empty() && !saveState(opt.outFile, state))
    {
        return 1;
    }

    double steps = (timedSteps > 0) ? (double)timedSteps : 1.0;
    double stepMs = 1000.0 * (forceTime + integrateTime) / steps;
    double forceMs = 1000.0 * forceTime / steps;
    double integrateMs = 1000.0 * integrateTime / steps;
    double interactionsPerSec = (forceTime > 0.0) ? (double)n * (double)n * (double)timedSteps / forceTime : 0.0;
    double gflops = interactionsPerSec * FLOPS_PER_INTERACTION * 1e-9;

    double energyDrift = 0.0;
    if(opt.energy)
    {
        Energy endEnergy = totalEnergy(state);
        energyDrift = (endEnergy.total() - startEnergy.total()) / fabs(startEnergy.total());
        printf("energy_start=%.10e,energy_end=%.10e,rel_drift=%.3e\n",
                startEnergy.total(), endEnergy.total(), energyDrift);
    }

    printf("label=%s,backend=%s,n=%d,steps=%llu,final_step=%llu,threads=%d,step_ms=%.3f,force_ms=%.3f,"
           "integrate_ms=%.3f,interactions_per_second=%.4e,gflops=%.2f\n",
           opt.label.c_str(), backendName(opt.backend), n, (unsigned long long)timedSteps,
           (unsigned long long)state.step, threads, stepMs, forceMs, integrateMs, interactionsPerSec, gflops);

    if(!opt.resultsFile.empty())
    {
        FILE* f = fopen(opt.resultsFile.c_str(), "a");
        if(f)
        {
            fprintf(f, "%s,%s,%d,%llu,%d,%.3f,%.3f,%.3f,%.4e,%.2f,%.3e\n",
            opt.label.c_str(), backendName(opt.backend), n, (unsigned long long)timedSteps,
            threads, stepMs, forceMs, integrateMs, interactionsPerSec, gflops, energyDrift);
            fclose(f);
        }
        else
        {
            fprintf(stderr, "Unable to open %s\n", opt.resultsFile.c_str());
        }
    }
    releaseKernel(resources);
    return 0;
}
