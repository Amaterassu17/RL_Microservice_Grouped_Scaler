#/!/bin/bash

sed -i -z 's|\(    \[plugins."io.containerd.grpc.v1.cri".registry\]\n      config_path = \)""|\1"/etc/containerd/certs.d"|g' /etc/containerd/config.toml
mkdir -p /etc/containerd/certs.d/docker.io
printf 'server = "https://registry-1.docker.io"\nhost."http://docker-cache.grid5000.fr".capabilities = ["pull", "resolve"]\n' | tee /etc/containerd/certs.d/docker.io/hosts.toml
systemctl restart containerd