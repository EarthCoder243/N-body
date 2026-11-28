#!/bin/sh
#Removes the stack (volumes and checkpoints are kept)
set -eu
docker stack rm nbody
