#CPU worker and supervisor image

FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt .
COPY core core
COPY apps apps

RUN cmake -S . -B build -DNBODY_GUI=OFF -DNBODY_MPI=ON -DNBODY_ARCH=x86-64-v3 \
    && cmake --build build -j

FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 openmpi-bin openssh-client openssh-server python3 \
    && rm -rf /var/lib/apt/lists/*

RUN userdel -r ubuntu \
    && groupadd --gid 1000 mpi \
    && useradd --create-home --uid 1000 --gid mpi --shell /bin/bash mpi \
    && usermod -p '*' mpi \
    && mkdir -p /opt/nbody /var/lib/nbody /ckpt /run/sshd \
    && rm -f /etc/ssh/ssh_host_*

RUN printf '%s\n' \
    'Port 22' \
    'PermitRootLogin no' \
    'PasswordAuthentication no' \
    'PubkeyAuthentication yes' \
    'AllowUsers mpi' \
    > /etc/ssh/sshd_config

COPY --from=builder /src/build/nbody-mpi /src/build/nbody-headless /src/build/nbody-compare /opt/nbody/
COPY deploy/entrypoint.sh /usr/local/sbin/entrypoint
COPY deploy/heartbeat.py deploy/supervisor.py /opt/nbody/

RUN sed -i 's/\r$//' /usr/local/sbin/entrypoint /opt/nbody/*.py \
    && chmod 0755 /usr/local/sbin/entrypoint

ENTRYPOINT ["/usr/local/sbin/entrypoint"]
