#!/bin/sh
#Fault injection for fault matrix. Usage: bench/faults.sh F1|F2|F4|F5|F6 [run_number]
#Appends time,fault,run,detail to bench/results/faults.csv
set -eu
FAULT="${1:?usage: faults.sh  F1|F2|F4|F5|F6 [run]}"
RUN="${2:-1}"
OUT="$(dirname "$0")/results/faults.csv"
mkdir -p "$(dirname "$OUT")"

record ()
{
    echo "$(date +%s.%3N),$FAULT,$RUN,$1" >> "$OUT"
    echo "$FAULT run $RUN: $1"
}

case "$FAULT" in
    F1) #Worker container crash, run on the node with target worker
        CONTAINER="$(docker ps -q -f name=nbody_worker-cpu | head -n 1)"
        [ -n "$CONTAINER" ] || { echo "No worker container on this node" >&2; exit 1; }
        record "docker kill $CONTAINER"
        docker kill "$CONTAINER" > /dev/null
        ;;
    F2|F5) #Node loss(F2 node3, F5 the supervisor's node)
        record "node $(hostname) going down"
        echo "Now shut this machine down (wsl --shutdown in PowerShell or power off)"
        ;;
    F4) #Straggler: burns every core on this node until stopped
        CORES="$(nproc)"
        record "burner start $CORES threads"
        docker run -d --rm --name nbody-burner busybox \
            sh -c "for i in \$(seq $CORES); do yes > /dev/null & done; wait" > /dev/null
        echo "Stop it with: docker stop nbody-burner"
        ;;
    F6) #Raft leader loss
        docker node ls >> "$OUT.nodes-before.txt"
        record "stopping docker on leader $(hostname)"
        sudo systemctl stop docker docker.socket
        echo "On another manager, run: docker node ls (new leader should appear within seconds)"
        ;;
    *)
        echo "Unknown fault $FAULT" >&2
        exit 1
        ;;
esac

