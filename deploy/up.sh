#!/bin/sh
#Deploys the stack. LOCAL=1 uses stack.local.yml
#JOB_ARGS, MIN_WORKERS, RANK0_NODE, CPU_THREADS are passed to stack.yml
set -eu
cd "$(dirname "$0")"
TAG="${TAG:-$(git rev-parse --short HEAD)}"
if [ -n "${REGISTRY:-}" ]; then
    IMAGE="$REGISTRY/nbody-cpu:$TAG"
else
    IMAGE="nbody-cpu:$TAG"
fi
export IMAGE
if [ "${LOCAL:-0}" = "1" ]; then
    docker stack deploy -c stack.yml -c stack.local.yml nbody
else
    docker stack deploy -c stack.yml nbody
fi
echo "Deployed $IMAGE, watch with: docker service logs -f nbody_supervisor"
