#/!/bin/bash

dir="$DIR/kube_cluster_prep"

git clone https://github.com/delimitrou/DeathStarBench.git
cp $dir/conf/socialnetwork/nginx-thrift-nodeport.yaml ./
cp $dir/conf/socialnetwork/media-frontend-nodeport.yaml ./

helm install socialnetwork ./DeathStarBench/socialNetwork/helm-chart/socialnetwork/
kubectl rollout status deployment nginx-thrift
kubectl apply -f nginx-thrift-nodeport.yaml
kubectl apply -f media-frontend-nodeport.yaml 

host_name=$(kubectl get pod --selector 'app=nginx-thrift' -o jsonpath='{.items[*].spec.nodeName}')
host_ip=$(kubectl get node $host_name -o jsonpath='{.status.addresses[0].address}')

echo "Social network accessible at $host_ip:30080"
