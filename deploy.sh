dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $dir
kube_dir=$dir/KubePrep/kube_cluster_prep
echo $kube_dir
kadeploy3 debian11-kube -f $OAR_FILE_NODES -k ~/.ssh/id_rsa.pub
cd $kube_dir
# chmod +x prepare.sh post_install.sh kube_setup_tools.sh
./prepare.sh 
./post_install.sh

CONTROL=$(head -n 1 $OAR_NODEFILE)
NODE=$(head -n 1 $OAR_NODEFILE)
WORKERS=$(tail -n +2 $OAR_NODEFILE | awk '!seen[$0]++' | grep -v "$NODE")

NUM_WORKERS=$(echo "$WORKERS" | wc -l)

# Create an array to store worker nodes
WORKER_NODES=()
for i in $(seq 1 $NUM_WORKERS); do
    WORKER_NODES+=($(echo "$WORKERS" | sed -n "${i}p"))
done

echo "$NODE + ${WORKER_NODES[0]}" # Just for debugging

export NODES="$(cat $OAR_FILE_NODES | uniq)"
for ((i = 0; i < $(wc -w <<<"$NODES"); i++ )); do
    if [ "$i" == 0 ]; then
        echo "$i ciao"
        ssh root@$CONTROL 'bash -s' < $kube_dir/kubeadm_init_control.sh $dir
        sleep 1 # corrected syntax for sleep
    else
        (
            echo "$i"
            ssh root@${WORKER_NODES[i-1]} "chmod +x $kube_dir/kubeadm_token.sh"
            ssh root@${WORKER_NODES[i-1]} 'bash -s' < $kube_dir/kubeadm_token.sh $dir
        ) &
        
    fi
done

ssh root@$CONTROL 'kubectl apply -f https://github.com/weaveworks/weave/releases/download/v2.8.1/weave-daemonset-k8s.yaml'
ssh root@$CONTROL "kubectl apply -f $dir/MetricsPrep/metric_server_components.yaml"
sleep 20
# ssh root@$CONTROL "chmod +x $dir/MetricsPrep/install_stack.sh"
ssh root@$CONTROL 'bash -s' < $dir/MetricsPrep/install_stack.sh $dir
ssh root@$CONTROL 'bash -s' < $dir/Istio_Mesh/install_istio_normal.sh "$dir/Istio_Mesh"
# ssh root@$CONTROL "chmod +x $dir/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh"
# ssh root@$CONTROL 'bash -s' < $dir/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh
# ssh root@$CONTROL 'chmod +x $dir/AWARE_mod/aware/multidimensional-pod-autoscaler/generate_recommender_dockerfile.sh'
# ssh root@$CONTROL 'bash -s' < $dir/AWARE_mod/aware/multidimensional-pod-autoscaler/generate_recommender_dockerfile.sh
# ssh root@$CONTROL 'chmod +x $dir/AWARE_mod/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh'
# ssh root@$CONTROL 'bash -s' < $dir/AWARE_mod/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh
# ssh root@$CONTROL 'chmod +x $dir/docker_gricad_registry_commands.sh'
ssh root@$CONTROL 'bash -s' < $dir/docker_gricad_registry_commands.sh
ssh root@$CONTROL "pip install -r $dir/rl_operator/rl/Cluster-Environment/cluster-environment/env/requirements.txt"
ssh root@$CONTROL 'pip install kopf kubernetes pyyaml pykube pick wandb'
ssh root@$CONTROL 'wandb login 6d649fa07ca77cbd798130b5467edae3c32b6690'
ssh root@$CONTROL 'apt install screen tmux -y'

