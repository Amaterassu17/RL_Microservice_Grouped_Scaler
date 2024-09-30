dir=$1/KubePrep/kube_cluster_prep
echo $dir
sed -i -z "s|\(    \[plugins.\"io.containerd.grpc.v1.cri\".registry\]\n      config_path = \)\"\"|\1\"/etc/containerd/certs.d\"|g" /etc/containerd/config.toml
mkdir -p /etc/containerd/certs.d/docker.io
printf "server = \"https://registry-1.docker.io\"\nhost.\"http://docker-cache.grid5000.fr\".capabilities = [\"pull\", \"resolve\"]\n" | tee /etc/containerd/certs.d/docker.io/hosts.toml
systemctl restart containerd
kubeadm join 172.16.39.3:6443 --token kzv3c6.0qr9ky7kbzbj61et --discovery-token-ca-cert-hash sha256:202e9fff7e64be616bc5f216681ebf9cec6d46861ac0e21377092af587c257fa 
