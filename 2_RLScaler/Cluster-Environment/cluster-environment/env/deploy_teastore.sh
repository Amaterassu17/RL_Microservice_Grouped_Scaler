#!/bin/bash
echo "Deploying the TeaStore application with ClusterIP configuration"
kubectl create -f https://raw.githubusercontent.com/DescartesResearch/TeaStore/master/examples/kubernetes/teastore-clusterip.yaml

#create all the deployment objects
#the MPA objects are for each deployment.apps
#auth db image persistence recommender registry webui
#apply kubectl with file in the directory deploy-i.yaml
# echo "Deploying the TeaStore application MPA"
# for i in auth db image persistence recommender registry webui
# do
#     kubectl apply -f ./deploy-mpa-$i.yaml
# done
