#Worker heartbeat: every second, tells supervisor overlay IP, node, backend, and thread count.
#Supervisor builds mpirun command from these, restarted containers are picked up automatically

import json
import os
import socket
import time

SUPERVISOR = os.environ.get("SUPERVISOR_ADDR", "supervisor")
PORT = int(os.environ.get("HEARTBEAT_PORT", "9999"))
NODE = os.environ.get("NODE_HOSTNAME", socket.gethostname())
BACKEND = os.environ.get("BACKEND", "omp")
THREADS = int(os.environ.get("THREADS", "0")) #0 = all cores the container sees
INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "1"))


def overlay_ip(address):
    #Connecting UDP socket, sends nothing, makes OS pick interface that reaches supervisor
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((address, PORT))
        return probe.getsockname()[0]
    finally:
        probe.close()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            address = socket.gethostbyname(SUPERVISOR)
            beat = {"ip": overlay_ip(address), "node":NODE, "backend":BACKEND,
                    "threads":THREADS, "task":socket.gethostname()}
            sock.sendto(json.dumps(beat).encode(), (address, PORT))
        except OSError:
            pass #Supervisor not up yet or moving
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
