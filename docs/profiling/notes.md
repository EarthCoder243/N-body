# Baseline profiling notes

## Setup
- Machine: MacBook (M5), Docker container ubuntu:24.04 (arm64), 10 cores
- Build: Release, -O3 -march=native -ffast-math, BENCH_ONLY=ON
- Galaxy init, seed 1, dt = 0.0035, eps2 = 40 (nbody::Params defaults)

## Where the time goes (gprof)
- Command: N-body-bench -backend seq -n 20000 -steps 5 -seed 1
- SeqKernel::compute: ___% of run time
- printEnergy (correctness check, runs twice): ___%
- Everything else (setup, leapfrog update): ~0%
- Full report: gprof_seq_n20000.txt
- perf was not used: it doesn't work inside Docker Desktop on macOS

## Correctness reference (energy and momentum)
- Command: N-body-bench -backend seq -n 2000 -steps 100 -seed 1
- Energy at step 0: ___
- Energy at step 100: ___
- Relative drift: ___
- Momentum at step 0 / step 100: ___ / ___
- Omp gives the same energy: yes / no

## Memory per particle
| Data | Bytes |
| --- | --- |
| State: pos + vel + acc (3 x 16 B) | 48 |
| Bench newAcc buffer | 16 |
| Omp kernel x/y/z/m copies (4 x 4 B) | 16 |
| Total | 80 |
| Sent per step by MPI_Allgatherv (positions only) | 16 |

- N = 1,000,000 needs about 80 MB, so memory is not the limit (3050 Ti has 4 GB)
- N = 200,000 sends about 3.2 MB per step over the network

## First kernel timings (N = 2000, Mac container; final numbers come from an x86 machine)
| Kernel | ms per step | Speedup |
| --- | --- | --- |
| Seq | 3.66 | 1x |
| Omp, 1 thread | 1.03 | 3.6x |
| Omp, 4 threads | 0.30 | 12.2x |