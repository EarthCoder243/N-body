#!/bin/sh
#Builds the image tagged with the git SHA, pushes it if REGISTRY is set
set -eu
cd "$(dirname "$0")/.."
TAG="$(git rev-parse --short HEAD)"
docker build -f deploy/Dockerfile -t nbody-cpu:"$TAG" -t nbody-cpu:latest .
if [ -n "${REGISTRY:-}" ]; then
    docker tag nbody-cpu:"$TAG" "$REGISTRY/nbody-cpu:$TAG"
    docker push "$REGISTRY/nbody-cpu:$TAG"
fi
echo "Built nbody-cpu:$TAG"
