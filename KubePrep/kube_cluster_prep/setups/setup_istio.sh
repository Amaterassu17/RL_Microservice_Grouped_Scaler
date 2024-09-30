#!/bin/bash


curl -L https://istio.io/downloadIstio | sh -
export PATH="$PATH:/root/istio-1.20.3/bin"
istioctl install -y --set components.egressGateways[0].name=istio-egressgateway --set components.egressGateways[0].enabled=true
kubectl label namespace default istio-injection=enabled


#source $DIR/kube_cluster_prep/setups/teastore_setup.sh
source $DIR/kube_cluster_prep/setups/socialnetwork_setup.sh

kubectl apply  -f $DIR/kube_cluster_prep/conf/istio/socialnetwork-gateway.yaml 

#kubectl create -f $DIR/kube_cluster_prep/conf/istio/gen.yaml 
#kubectl apply -f $DIR/kube_cluster_prep/conf/istio/socialnetwork-gateway.yaml 
#kubectl apply -f $DIR/kube_cluster_prep/conf/istio/gen-gateway-socialnetwork.yaml

#/home/ggrabher/kube_cluster_prep/setups/teastore_setup.sh 
#istioctl analyze
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')
export SECURE_INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="https")].nodePort}')
export INGRESS_HOST=$(kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}')
export GATEWAY_URL=$INGRESS_HOST:$INGRESS_PORT
echo "http://$GATEWAY_URL/"
