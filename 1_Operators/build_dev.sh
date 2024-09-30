dir=$(pwd)
docker login gricad-registry.univ-grenoble-alpes.fr -u <username> -p <password>
docker build -t gricad-registry.univ-grenoble-alpes.fr/microserviceserods/aware-deployment-grid5k/rl-operator:dev -f Dockerfile_dev $dir
docker push gricad-registry.univ-grenoble-alpes.fr/microserviceserods/aware-deployment-grid5k/rl-operator:dev