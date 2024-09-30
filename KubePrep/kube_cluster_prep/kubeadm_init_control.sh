dir=$1/KubePrep/kube_cluster_prep
echo "kubeadm_init_control $dir"

sed -i -z 's|\(    \[plugins."io.containerd.grpc.v1.cri".registry\]\n      config_path = \)""|\1"/etc/containerd/certs.d"|g' /etc/containerd/config.toml
mkdir -p /etc/containerd/certs.d/docker.io
printf 'server = "https://registry-1.docker.io"\nhost."http://docker-cache.grid5000.fr".capabilities = ["pull", "resolve"]\n' | tee /etc/containerd/certs.d/docker.io/hosts.toml
systemctl restart containerd



kubeadm init --config ./kubeadm-config.yaml
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
echo 'dir=$1/KubePrep/kube_cluster_prep' > $dir/kubeadm_token.sh
echo 'echo $dir' >> $dir/kubeadm_token.sh
echo 'sed -i -z "s|\(    \[plugins.\"io.containerd.grpc.v1.cri\".registry\]\n      config_path = \)\"\"|\1\"/etc/containerd/certs.d\"|g" /etc/containerd/config.toml' >> $dir/kubeadm_token.sh
echo 'mkdir -p /etc/containerd/certs.d/docker.io' >> $dir/kubeadm_token.sh
echo 'printf "server = \"https://registry-1.docker.io\"\nhost.\"http://docker-cache.grid5000.fr\".capabilities = [\"pull\", \"resolve\"]\n" | tee /etc/containerd/certs.d/docker.io/hosts.toml' >> $dir/kubeadm_token.sh
echo 'systemctl restart containerd' >> $dir/kubeadm_token.sh
kubeadm token create --print-join-command >> $dir/kubeadm_token.sh