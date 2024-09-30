dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
num_nodes=$1
echo "numnodes are $num_nodes"

echo "Undeploying the TeaStore"
kubectl patch deployments teastore-auth teastore-db teastore-image teastore-persistence teastore-image teastore-webui teastore-recommender teastore-registry -p '{"metadata": {"finalizers": []}}' --type merge
kubectl patch deployments mpa-operator -p '{"metadata": {"finalizers": []}}' --type merge
$dir/../../TeaStore_deploy/undeploy_teastore.sh

i=3
for f in $dir/experiments_group/*; do
    if [ -f "$f" ]; then

        if [ $i -gt 4 ];
        then
            break
        fi

        echo "Deploying the TeaStore"
        cd /home/agennuso/aware-deployment-grid5k/rl_operator/times_experiments/
        dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
        $dir/../../TeaStore_deploy/deploy_teastore.sh

        sleep 30
        echo "Running experiment ex$i"
        export GROUP_FILE="$dir/experiments_group/ex$i.txt"
        export NUM_NODES=$num_nodes
        kopf run hooks_scaling_times_experiments.py --dev --standalone -n default
        cat $GROUP_FILE
        echo ""
        i=$((i+1))
        sleep 30
        echo "Undeploying the TeaStore"
        kubectl patch deployments teastore-auth teastore-db teastore-image teastore-persistence teastore-image teastore-webui teastore-recommender teastore-registry -p '{"metadata": {"finalizers": []}}' --type merge
        kubectl patch deployments mpa-operator -p '{"metadata": {"finalizers": []}}' --type merge
        $dir/../../TeaStore_deploy/undeploy_teastore.sh
    fi
done

sleep 10000

