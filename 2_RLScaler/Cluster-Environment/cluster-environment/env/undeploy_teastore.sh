echo "Undeploying the TeaStore application with ClusterIP configuration"
kubectl delete -f https://raw.githubusercontent.com/DescartesResearch/TeaStore/master/examples/kubernetes/teastore-clusterip.yaml
