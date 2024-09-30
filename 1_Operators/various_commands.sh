
#for the stable build

sleep 5

kubectl apply -f $dir/mpa_operator.yaml

sleep 15

# #to pull
# docker login gricad-registry.univ-grenoble-alpes.fr -u "gennusoa" -p "StspaNyungEXhqjDZEdC"
# docker pull gricad-registry.univ-grenoble-alpes.fr/microserviceserods/aware-deployment-grid5k/rl-operator:latest

# #create container and ssh into it
# docker run -it gricad-registry.univ-grenoble-alpes.fr/microserviceserods/aware-deployment-grid5k/rl-operator:latest /bin/bash

# #ssh into the pod
kubectl apply -f /home/agennuso/Boh/rl_operator/mpa_operator.yaml
sleep 20
export MPA_OPERATOR_POD=$(kubectl get pods -n default | grep mpa-operator | awk '{print $1}')
kubectl exec -it $MPA_OPERATOR_POD -- /bin/bash

#if pvc is not terminating
kubectl patch pvc pv-claim-name -p '{"metadata":{"finalizers":null}}'
kubectl patch mpa-operator -p '{"metadata": {"finalizers": []}}' --type merge
kubectl delete pv (pv name) --grace-period=0 --force



# #crictl pull gricad-registry.univ-grenoble-alpes.fr/microserviceserods/aware-deployment-grid5k/rl-operator --creds gennusoa:StspaNyungEXhqjDZEdC not working