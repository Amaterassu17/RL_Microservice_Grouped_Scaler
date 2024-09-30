dir=$1/MetricsPrep
kubectl create namespace monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install -f $dir/helm_values_stack.yaml prometheus prometheus-community/kube-prometheus-stack
export GRAFANA_SERVICE=$(kubectl get services -n default | grep grafana | awk '{print $1}')
kubectl patch svc $GRAFANA_SERVICE -n default --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
export GRAFANA_POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana" -o jsonpath="{.items[0].metadata.name}")
export PROMETHEUS_SERVICE=$(kubectl get services -n monitoring | grep prometheus-kube-prometheus-prometheus | awk '{print $1}')
kubectl patch svc $PROMETHEUS_SERVICE -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
export PROMETHEUS_PORT=$(kubectl get services $PROMETHEUS_SERVICE -n monitoring -o wide | tail -n 1 |awk '{print $5}' | cut -d':' -f2 | cut -d'/' -f1)
export CONTROL_IP=$(kubectl get nodes -o wide | grep control-plane | awk '{print $6}')
export GRAFANA_SVC_PORT=$(kubectl get services -l "app.kubernetes.io/name=grafana" -n default -o wide | tail -n 1 |awk '{print $5}' | cut -d':' -f2 | cut -d'/' -f1)
export GRAFANA_HOST="http://$CONTROL_IP:$GRAFANA_SVC_PORT"
export PROM_HOST="http://$CONTROL_IP:$PROMETHEUS_PORT"
echo $GRAFANA_HOST
echo $PROM_HOST
kubectl apply -f $dir/prometheus_service_temp.yaml