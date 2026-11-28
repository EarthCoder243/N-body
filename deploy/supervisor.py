#Supervisor for the N-Body job on Swarm. Tracks live workers from UDP heartbeats, launches one mpirun across them.
#Relaunches with --resume when a worker is lost, mpirun fails, or a new worker joins.
#Event written as time,event,detail to supervisor-events.csv

import json
import os
import signal
import socket
import subprocess
import threading
import time

HEARTBEAT_PORT = int(os.environ.get("HEARTBEAT_PORT", "9999"))
MIN_WORKERS = int(os.environ.get("MIN_WORKERS", "2")) #Won't launch with fewer
DEAD_AFTER = float(os.environ.get("DEAD_AFTER", "5")) #Seconds of silence before worker is lost
SETTLE = float(os.environ.get("SETTLE", "5")) #Worker set must stay the same this long before launch
SUBNET = os.environ.get("MPI_SUBNET", "10.20.0.0/24")
JOB_ARGS = os.environ.get("JOB_ARGS", "-n 20000 --steps 1000 --ckpt-every 50").split()
RANK0_NODE = os.environ.get("RANK0_NODE", "") #Node whose worker becomes rank 0 
SCALE_OUT = os.environ.get("SCALE_OUT", "1") == "1" #Relaunch to include newly joined worker
PROGRESS_EVERY = os.environ.get("PROGRESS_EVERY", "10")
BINARY = "/opt/nbody/nbody-mpi"
CKPT_DIR = "/var/lib/nbody/ckpt" #Each container's entrypoint points this at its own folder
EVENTS_FILE = os.path.join(CKPT_DIR, "supervisor-events.csv")

workers = {} #Overlay IP -> {"node", "backend", "threads", "last"}
lock = threading.Lock()

def log_event(event, detail=""):
    line = "%.3f,%s,%s" % (time.time(), event, detail)
    print(line, flush=True)
    try:
        with open(EVENTS_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass

def listen_heartbeats():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", HEARTBEAT_PORT))
    while True:
        data, sender = sock.recvfrom(4096)
        try:
            beat = json.loads(data.decode())
        except ValueError:
            continue
        ip = beat.get("ip", sender[0])
        with lock:
            if ip not in workers:
                log_event("worker_seen", "%s %s %s" % (ip, beat.get("node"), beat.get("backend")))
            workers[ip] = {"node": str(beat.get("node", ip)), "backend": str(beat.get("backend", "omp")),
                           "threads": int(beat.get("threads", 0)), "last": time.time()}

def live_workers():
    #Drops workers that have gone quiet
    now = time.time()
    with lock:
        for ip in [ip for ip, w in workers.items() if now - w["last"] > DEAD_AFTER]:
            log_event("worker_lost", "%s %s" % (ip, workers[ip]["node"]))
            del workers[ip]
        return {ip: dict(w) for ip, w in workers.items()}

def wait_for_workers():
    #Waits until at least MIN_WORKERS are alive and the set hasn't changed for SETTLE seconds
    last_set = None
    stable_since = time.time()
    while True:
        live = live_workers()
        current = set(live)
        if current != last_set:
            last_set = current
            stable_since = time.time()
        elif len(current) >= MIN_WORKERS and time.time() - stable_since >= SETTLE:
            return live
        time.sleep(0.5)

def worker_order(item):
    #Sort key : RANK0_NODE's worker first, then by name, then by IP
    ip, w = item
    return (w["node"] != RANK0_NODE, w["node"], ip)

def build_command(live):
    #One app context per worker, each rank gets own backend and thread count
    order = sorted(live.items(), key=worker_order)
    cmd = ["mpirun", "--bind-to", "none",
           "--mca", "pml", "ob1", "--mca", "btl", "tcp,self",
           "--mca", "btl_tcp_if_include", SUBNET, "--mca", "oob_tcp_if_include", SUBNET,
           "--mca", "plm_rsh_no_tree_spawn", "1"]
    for i, (ip, w) in enumerate(order):
        if i > 0:
            cmd.append(":")
        cmd += ["-np", "1", "-H", ip + ":1", BINARY, "--backend", w["backend"]]
        if w["threads"] > 0:
            cmd += ["--threads", str(w["threads"])]
        cmd += JOB_ARGS + ["--ckpt-dir", CKPT_DIR, "--resume", "--progress", PROGRESS_EVERY]
    return cmd, order

def relay_output(proc):
    #Passes mpirun output to the service log and timestamps the lines used for recovery times
    for line in proc.stdout:
        print(line, end="", flush=True)
        if line.startswith("progress"):
            log_event("progress", line.strip())
        elif line.startswith("Resumed") or line.startswith("No checkpoints"):
            log_event("resume", line.strip())
        elif line.startswith("rebalance"):
            log_event("rebalance", line.strip().replace(",", " "))

def stop(proc):
    #Stops mpirun and its ssh children, remote ranks exit when ssh closes
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

def run_job(live):
    cmd, order = build_command(live)
    log_event("launch", "%d workers %s" % (len(order), " ".join(w["node"] + "/" + w["backend"] for ip, w in order)))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    reader = threading.Thread(target=relay_output, args=(proc, ), daemon=True)
    reader.start()

    result = None
    while proc.poll() is None:
        time.sleep(0.5)
        current = live_workers()
        lost = [ip for ip in live if ip not in current]
        joined = [ip for ip in current if ip not in live]
        if lost:
            result = "lost"
            log_event("stop_job", "lost " + " ".join(lost))
            break
        if SCALE_OUT and joined:
            result = "scale_out"
            log_event("stop_job", "joined " + " ".join(joined))
            break
    if result is not None:
        stop(proc)
    reader.join(timeout=5)
    log_event("exit", str(proc.returncode))
    if result is None:
        result = "done" if proc.returncode == 0 else "failed"
    return result

def main():
    threading.Thread(target=listen_heartbeats, daemon=True).start()
    log_event("supervisor_start", socket.gethostname())
    while True:
        live = wait_for_workers()
        if run_job(live) == "done":
            log_event("job_done")
            while True: #Stays up so logs remain, restart would just resume finished job and exit
                time.sleep(3600)
        time.sleep(1)

if __name__ == "__main__":
    main()
