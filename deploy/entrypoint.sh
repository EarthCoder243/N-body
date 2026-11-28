#!/bin/sh
#Container start for both roles. Installs the MPI SSH key from Docker secrets for user mpi,
#points /var/lib/nbody/ckpt at this containers own folder in the volume, then starts either the supervisor or the heartbeat + sshd
set -eu

ROLE="${ROLE:-worker}"
NODE="${NODE_HOSTNAME:-$(hostname)}"
SLOT="${TASK_SLOT:-0}"

if [ ! -s /run/secrets/mpi_ssh_key ] || [ ! -s /run/secrets/mpi_ssh_pub ]; then 
    echo "MPI SSH secrets are missing; create mpi_ssh_key and mpi_ssh_pub first." >&2
    exit 1
fi

install -d -o mpi -g mpi -m 0700 /home/mpi/.ssh
install -o mpi -g mpi -m 0600 /run/secrets/mpi_ssh_key /home/mpi/.ssh/id_ed25519
install -o mpi -g mpi -m 0600 /run/secrets/mpi_ssh_pub /home/mpi/.ssh/authorized_keys
cat > /home/mpi/.ssh/config << 'EOF'
Host *
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
EOF
chown mpi:mpi /home/mpi/.ssh/config
chmod 0600 /home/mpi/.ssh/config

#Own folder per container
install -d -o mpi -g mpi -m 0755 "/ckpt/$NODE-$SLOT"
ln -sfn "/ckpt/$NODE-$SLOT" /var/lib/nbody/ckpt

if [ "$ROLE" = "supervisor" ]; then
    exec runuser -u mpi -- python3 /opt/nbody/supervisor.py
fi

ssh-keygen -A
install -d -m 0755 /run/sshd
runuser -u mpi -- python3 /opt/nbody/heartbeat.py &
exec /usr/sbin/sshd -D -e

