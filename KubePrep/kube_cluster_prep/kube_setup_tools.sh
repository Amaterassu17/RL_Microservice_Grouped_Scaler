dir=$(pwd)/../..

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
        ssh root@$CONTROL 'bash -s' < "cd $dir; ./KubePrep/kube_cluster_prep/kube_setup_tools.sh"
        sleep 1 # corrected syntax for sleep
    else
        echo "$i"
        ssh root@${WORKER_NODES[i-1]} "chmod +x $dir/KubePrep/kube_cluster_prep/kubeadm_token.sh"
        ssh root@${WORKER_NODES[i-1]} 'bash -s' < "cd $dir; ./KubePrep/kube_cluster_prep/kubeadm_token.sh"
    fi
done

ssh root@$CONTROL 'kubectl apply -f https://github.com/weaveworks/weave/releases/download/v2.8.1/weave-daemonset-k8s.yaml'
ssh root@$CONTROL "kubectl apply -f $dir/MetricsPrep/metric_server_components.yaml"
sleep 20
ssh root@$CONTROL "chmod +x $dir/MetricsPrep/install_stack.sh"
ssh root@$CONTROL 'bash -s' < "cd $dir; $dir/MetricsPrep/install_stack.sh"
# ssh root@$CONTROL "chmod +x $dir/AWARE_mod/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh"
# ssh root@$CONTROL "$dir/AWARE_mod/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh"
# ssh root@$CONTROL 'chmod +x /home/agennuso/Boh/AWARE_mod/aware/multidimensional-pod-autoscaler/generate_recommender_dockerfile.sh'
# ssh root@$CONTROL 'bash -s' < /home/agennuso/Boh/AWARE_mod/aware/multidimensional-pod-autoscaler/generate_recommender_dockerfile.sh
# ssh root@$CONTROL 'chmod +x /home/agennuso/Boh/AWARE_mod/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh'
# ssh root@$CONTROL 'bash -s' < /home/agennuso/Boh/AWARE_mod/aware/multidimensional-pod-autoscaler/deploy/mpa-up.sh
