dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Deploying the TeaStore"
$dir/../../TeaStore_deploy/deploy_teastore.sh

sleep 30

i=1
for f in $dir/experiments_group/*; do
    if [ -f "$f" ]; then
        echo "Running experiment ex$i"
        export GROUP_FILE="$dir/experiments_group/ex$i.txt"
        export NUM_NODES=2
        kopf run hooks_scaling_times_experiments.py --dev --standalone -n default
        cat $GROUP_FILE
        echo ""
        i=$((i+1))
        sleep 30
    fi
done

echo "Undeploying the TeaStore"
kubectl patch deployments teastore-auth teastore-db teastore-image teastore-persistence teastore-image teastore-webui teastore-recommender teastore-registry -p '{"metadata": {"finalizers": []}}' --type merge
kubectl patch deployments mpa-operator -p '{"metadata": {"finalizers": []}}' --type merge
$dir/../../TeaStore_deploy/undeploy_teastore.sh
