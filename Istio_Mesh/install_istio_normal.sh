this_dir=$1
dir=$1/istio-1.21.0/bin
echo $dir
echo $this_dir
export PATH=$dir:$PATH
CONTROL_IP=$(kubectl get nodes -o wide | grep control-plane | awk '{print $6}')

# #to install istio
istioctl install --set profile=demo -y

# #inject Envoy automatically on each app deployed in default
kubectl label namespace default istio-injection=enabled

# #if want to apply to everything but there are warnings
kubectl rollout restart deployment

#since we don't have an external LoadBalancer
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')
export SECURE_INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="https")].nodePort}')
export INGRESS_HOST=$(kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}')
export GATEWAY_URL="http://$INGRESS_HOST:$INGRESS_PORT"

dir=$(pwd)/istio-1.21.0/samples/addons

kubectl apply -f $dir/jaeger.yaml
kubectl apply -f $this_dir/prometheus-scrape-istio.yaml

# kubectl delete -f /home/agennuso/Boh/Istio_Mesh/istio-1.21.0/samples/addons/kiali.yaml
# kubectl apply -f $dir/kiali.yaml

# kubectl patch service kiali -n istio-system --type='json' -p='[{"op": "replace", "path": "/spec/type", "value": "NodePort"}]'
# export KIALI_PORT=$(kubectl get service kiali -n istio-system -o wide | tail -n 1 |awk '{print $5}' | cut -d':' -f2 | cut -d'/' -f1)
# export KIALI_URL="http://$CONTROL_IP:$KIALI_PORT"
# kubectl rollout status deployment/kiali -n istio-system
# echo $KIALI_URL


# kubectl delete -f /home/agennuso/Boh/Istio_Mesh/istio-1.21.0/samples/addons/jaeger.yaml
# kubectl patch service tracing -n istio-system --type='json' -p='[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
# export JAEGER_PORT=$(kubectl get service tracing -n istio-system -o wide | tail -n 1 |awk '{print $5}' | cut -d':' -f2 | cut -d'/' -f1)
# export JAEGER_URL="http://$CONTROL_IP:$JAEGER_PORT"
# echo $JAEGER_URL