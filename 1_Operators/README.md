# Coordinated Scaling Operator.

This folder contains all the files to modify, build and deploy The Coordinated Scaling Operator.
The Operator, for now, will just scale the groups randomly every 60 seconds.



## A bit of Theory behind it

The following Operator looking at what a well-designed Operator should do in a Kubernetes Environment.
The focus revolves around the usage of the *control loop pattern*

More info on https://kubernetes.io/docs/concepts/extend-kubernetes/operator/

## How it works for now

The Operator main objective now is to implement Coordinated Scaling through the usage of Python and the Kubernetes API.
The Operator will be able to scale groups of microservices together, and it will be able to do it in a coordinated way.

For now the groups are defined in a static way thanks to a group file.
The Operator will read the file and will scale the groups of microservices together.


# Structure of the folder


```bash
.
├── build_dev.sh # Script to build the operator in a development environment
├── build_stable.sh # Script to build the operator in a stable environment
├── Dockerfile # Dockerfile to build the operator
├── Dockerfile_dev # Dockerfile to build the operator in a development environment
├── mpa_operator_dev.yaml # Operator deployment file for development
├── mpa_operator.yaml # Operator deployment file for stable
├── pvc_debug # Folder in which the dev files are kept. Modify this if you want to change the operator WHILE it is deployed in a POD
│   ├── groups_1.txt
│   ├── groups.txt # Statically defined groups for microservices. Modify this if you want to change the groups
│   ├── hooks_horizontal.py # The main file of the operator. Modify this if you want to change the operator. Contains all the handlers for the oprerator events
│   ├── main.py
│   ├── prometheus_adaptor.py
│   ├── roundTime.py
│   └── util.py # Utility functions and classes for the operator
├── README.md # This file ;)
├── src # Folder in which the stable build files are kept. Try to modify this as less as possible.
│   ├── groups.txt
│   ├── handlers.py
│   ├── hooks_horizontal.py
│   ├── hooks.py
│   ├── __pycache__
│   │   └── hooks.cpython-39.pyc
│   └── requirements.txt
├── try_horizontal_autoscaler.py # Script to test some functions before putting them in the operator
├── try_scale.py
└── various_commands.sh # Some useful commands to interact with the operator
    
```

# Tutorial

In the following tutorial, Step 3 and Step 4 are interchangeable. You can deploy the Operator first and then the App or viceversa.
Deploying the Operator first will allow you to see the logs of the Operator and see if it is working properly before actually deploying the App. It is useful in those cases in which one would like to modify parts of the code that are not directly related to the App that will be deployed (e.g. Startup, Login on the cluster, Shutdown of the Operator etc...)

That is how I structure the Tutorial. One could also do viceversa and deploy the Operator after the App.

## 1. Deploy a Kubernetes Cluster

Make sure to setup a Kubernetes Cluster. You can use Minikube, Kind or any other Kubernetes Cluster.
In my deployments, I have always used GRID5000.

As of now its usage is not implemented yet, but the cluster will require a working Prometheus instance to retrieve the metrics at some point.
I deploy Prometheus usign the helm-chart provided in the Prometheus Community on GitHub.

https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack


## 2. Setup the App

Choose beforehand your application

Your application should have in all the Deployments section of the .yaml files a metadata field that is equal to the app_name that you will pass as an environment variable to the Operator.
The application should be run in a namespace that will also be setup in the Operator through evnironment variables.

Reasonably, the dynamic group suddivision still has to be defined, so we will use a group.txt file to define the groups of microservices that we want to scale together.

### 2.1 Define the groups

Inside the folder *pvc_debug* there is a file called *groups.txt* (also in the src folder but reasonably don't modify it before having something definitive that you want to build using the definitive Dockerfile).
This file contains the groups of microservices that you want to scale together.

The file is structured in lines, each line is a group of microservices that you want to scale together.
For example, we use TeaStore (which is also is the default app_name that is used in the Operator if not defined in the environment) and we want to scale the webui, the db and the image microservices together.

The file will look like this:

```bash
teastore-webui,teastore-db,teastore-image
other_microservices
other_microservices
```



## 3. Choose how to deploy the Operator.

The Operator can be deployed in three different ways as of now.

### 3.1 By Running the scripts directly

The first one is by running the script directly.
The script can be found either in the ./pvc_debug or in the src folder.
It is advised to use the *hooks_horizontal.py* file as is the one that is already setup to work in a Kubernetes environment.

#### 3.1.1 Setup Environment Variables

The script requires some *Environment Variables* to be used properly.
Namely:
- *GROUP_FILE*: The file in which the groups are defined. The file should be in the same folder as the script. The default value is "./groups.txt"
- *NAMESPACE*: The namespace in which the application is deployed. The default value is "default"
- *APP_NAME*: The name of the application (e.g. for TeaStore it will be teastore). The default value is "teastore"
- *KUBERNETES_POD*: A boolean value that tells the script if it is running in a Kubernetes environment or not. If you decide to set it up and use this script in a script without POD, set it to False. Ths is due t a need for a different configuration of the kubernetes client in the script. If by chance this is setup and is "true" the script will crash.

#### 3.1.2 Install the required Python Packages

The script requires some Python packages to be installed. They can be found inside the folder *pvc_debug* in the file *requirements.txt*.

Just run:

```bash
pip install -r requirements.txt
```

The packages are the following:
- kubernetes
- kopf
- pykube
- pyyaml

Particularly, the *kopf* package is the one that is used to run the Operator.

#### 3.1.3 Run the script

This step can also be done after the next one, the order is indifferent.
I've decided to put this before because it is easier to visualize the first logs of the Operator and see that it is working properly.

cd into the .pvc_debug folder and run the following command:

```bash
kopf run hooks_horizontal.py -n default --dev
```

What this command does is to run the Operator in a development mode, therefore it sets its priority to the maximum possible.
It also should be noted that in this case the Operator will be able just to operate on the default namespace. (It is advised to be coherent between this flag and the environment variable that needs to be defined)
In this mode it is possible to see both the logs provided by the logging Python package and the prints present in the file.


#### 3.1.4. Deploy an App

*Skip directly to 4. Deploy an App*

### 3.2 By Building the Docker Image and Deploying it in a Kubernetes Cluster as a POD (DEV)


To deploy the Operator inside a POD in DEV mode means deploying the Operator .yaml file with everything setup inside it (RBAC and PVs), so that you can modify the Operator/files in the POD and then run it.

#### 3.2.1 A look into the Dockerfile and the YAML file

##### Dockerfile

The Dockerfile is used to build an image of this application.
In this very specific case we will use the Dockerfile_dev file to build the image. Inside there are some lines that are commented out, but the substance is there. I will now report the whole file explaining what each line does.

Since we will be using a Persistent Volume to store the groups.txt file, we will need to mount it in the POD. This is done by creating a Persistent Volume Claim and a Persistent Volume in the Kubernetes Cluster. The Persistent Volume Claim will be mounted in the POD, so we don't need to copy the folders in the image, as the image will just mount the volume and read the files that can be modified on the go.
That's the objective of having a Persistent Volume (can be found in the yaml file)

```Dockerfile
# We use the python 3.11 slim image as a base, can upgrade to every python image you want. ATTENTION: Normal tags of Python are pretty heavy so debugging and building is not advised with them because they will be very heavy (~300MB) compared to the slim version (~70MB)
FROM python:3.11-slim
# SETUP ENVIRONMENT VARIABLES
ENV KUBERNETES_POD true
ENV GROUP_FILE ./groups.txt
ENV APP_NAME teastore
ENV NAMESPACE default

RUN pip install kopf kubernetes pyyaml pykube prometheus_client --no-cache-dir


# We run a sleep loop since we want to get inside the pod through ssh and modify the files on the go. We can run kopf run... as always from inside the pod
CMD ["/bin/bash", "-c", "while true; do sleep 30; done;"]
```

Once the image is built, it should be pushed to a Docker Registry and then deployed in the Kubernetes Cluster using the yaml file.
For now, it is setup to push everything on the Container Registry on my GitLab account.

I advise either to setup a container registry on GitLab or use Docker + some kind of cache to not reach the pull limit of DockerHub.

A possible way of doing it is by using the following (opportunely modified) commands:

```bash
dir=$(pwd)
docker login <repo> -u <username> -p <password>
docker build -t <repo>/<image-name>:<tag> -f Dockerfile_dev $dir
docker push <repo>/<image-name>:<tag>
```

If you are affiliated with the gricad.gitlab repository, you should be able to use the commands found in the *build_dev.sh* script.
Just login, build and push the image. I would advise you to push images with different names so that it won't overwrite other's people eventual work.
Finally, to be able to pull from that registry, you should create a secret in the Kubernetes Cluster. The following command should be used:

```bash
kubectl create secret docker-registry gricad-registry --docker-server=https://gricad-registry.univ-grenoble-alpes.fr --docker-username=<username> --docker-password=<access-token>
```

The docker username and docker password must be setup by a manager of the repository. Refer to Vania Marangozova or Albin Petit for it.

If you are not affiliated with the gricad.gitlab repository, then you should setup your own container registry and push the image there or use DockerHub. I will not provide a tutorial for that.

I advise to use the tag *dev* or something else that is not either *stable* or *latest* for the image.

The image should be now pushed to the registry. Now we need to tune the yaml file so that it uses the correct image.

##### YAML file

The YAML file is the file that will be used to deploy the Operator in the Kubernetes Cluster.
It contains some RBAC so that the POD can execute python scripts that can
1. Interact with the Kubernetes API
2. Read, Modify and Delete Kubernetes Objects

They are put in the top of the file.

Then the PersistentVolume and the PersistentVolumeClaims are created. They will take everything that is contained in the *pvc_debug* folder and mount it in the POD.

Finally the Deployment is created. The Deployment will use the image that we have just pushed to the registry and will mount the PersistentVolumeClaim in the /app folder of the POD.
I highly suggest

Now that we have everything setup, we can actually deploy the pod in the Kubernetes Cluster, all we need is the following code:

```bash
kubectl apply -f mpa_operator_dev.yaml
```

We can check that is running by running the following command:

```bash
kubectl get pods -A
``` 

The container will now be executing a sleep command in a while loop. What we want to do now is to get inside the container and run the Operator.

Sorry about the unequal name, this will undergo a name change soon.
```bash
export MPA_OPERATOR_POD=$(kubectl get pods -n default | grep mpa-operator | awk '{print $1}')
kubectl exec -it $MPA_OPERATOR_POD -- /bin/bash
```

We can run the script already since we have every ENV variable and every package installed.
Now that we are inside the container, we should cd to the pvc_debug folder and run the following command:

```bash
cd pvc_debug
kopf run hooks_horizontal.py -n default --dev
```


#### 3.2.2 Deploy the App

*Skip directly to 4. Deploy the App*


### 3.3 By Building the Docker Image and Deploying it in a Kubernetes Cluster as a POD (STABLE)

The stable version of the Operator is the one that is used in production.
All the files are present in the ./src folder.

#### 3.3.1 A look into the Dockerfile and the YAML file

The Dockerfile is used to build an image of this application.
In this very specific case we will use the Dockerfile to build the image. Inside there are some lines that are commented out, but the substance is there. I will now report the whole file explaining what each line does.

Unlike the DEV version, the Kubernetes Deployment will not be accompanied (for now at least) by a Persistent Volume and a Persistent Volume Claim. That means that everything that is there will be copied in the image and will be static. To manage eventual output files, it is still in the plans to use a Persistent Volume Claim, but for now there are none so the files will be static.

```Dockerfile
# We use the python 3.11 slim image as a base, can upgrade to every python image you want. ATTENTION: Normal tags of Python are pretty heavy so debugging and building is not advised with them because they will be very heavy (~300MB) compared to the slim version (~70MB)
FROM python:3.11-slim
# SETUP ENVIRONMENT VARIABLES
ENV KUBERNETES_POD true
ENV GROUP_FILE ./groups.txt
ENV APP_NAME teastore
ENV NAMESPACE default

RUN pip install kopf kubernetes pyyaml pykube prometheus_client --no-cache-dir


# We directly run the script since we don't need to modify anything on the go.
CMD ["kopf", "run", "hooks_horizontal.py", "-n", "default", "--dev", "--standalone]
```

To build the image and push it to the registry, the same commands as before can be used.
Once the image is built, it should be pushed to a Docker Registry and then deployed in the Kubernetes Cluster using the yaml file.
For now, it is setup to push everything on the Container Registry on my GitLab account.

I advise either to setup a container registry on GitLab or use Docker + some kind of cache to not reach the pull limit of DockerHub.

A possible way of doing it is by using the following (opportunely modified) commands:

```bash
dir=$(pwd)
docker login <repo> -u <username> -p <password>
docker build -t <repo>/<image-name>:<tag> -f Dockerfile_dev $dir
docker push <repo>/<image-name>:<tag>
```

I advise to use the tag *stable* or *latest* for the image.

The image should be now pushed to the registry. Now we need to tune the yaml file so that it uses the correct image.

##### YAML file

The YAML file is the file that will be used to deploy the Operator in the Kubernetes Cluster.
It contains some RBAC so that the POD can execute python scripts that can
1. Interact with the Kubernetes API
2. Read, Modify and Delete Kubernetes Objects

They are put in the top of the file.

Finally the Deployment is created. The Deployment will use the image that we have just pushed to the registry and will mount the PersistentVolumeClaim in the /app folder of the POD.

Now that we have everything setup, we can actually deploy the pod in the Kubernetes Cluster, all we need is the following code:

```bash
kubectl apply -f mpa_operator.yaml
```

We can check that is running by running the following command:

```bash
kubectl get pods -A
``` 


#### 3.3.2 Deploy the App

*Skip directly to 4. Deploy the App*

## 4. Deploy the App

Now if everything is setup correctly, one should deploy an application in the chosen namespace and with every microservice that has a metadata field that is equal to the app_name passed as environment variable.

For reference, we deploy TeaStore which is pretty easy to deploy and has precisely 7 microservices.
The microservices are deployed as Deployments in the Kubernetes environment and should be Running at the end.

Specifically for TeaStore, they are:

- teastore-webui
- teastore-db
- teastore-image
- teastore-persistence
- teastore-recommender
- teastore-registry
- teastore-auth


To deploy TeaStore, you can use the script I provided in the root of the whole repository, specifically the *deploy_teastore.sh* script found in the <root_repo>/TeaStore_deploy folder.


## 5. Test the Operator

Now the Operator should be running. If executed in script in place or within a POD, prints will be available (logging still has to be implemented) and the Operator should be able to scale the groups of microservices together.

Specifically, one can find inside the *utils.py* file all the variables that define the scaling. For now, just the *ACTION_INTERVAL* variable is defined, that defines the time interval between two scaling actions.

For now the implemented scaling is random, but it is possible to implement a more sophisticated scaling algorithm.


## 6. Scaling Times Experiments

I am also providing the files used to test the scaling times of the Operator.
Specifically, there is a directory called times_experiments that contains the same files as pvc_debug, without the integration of the RL logic and just useful to test various groups and various scalings.
Let's say one wants to test what is the total scaling time of one group of microservices, one can use the operator present in the folder with the dev_pod configuration.

The total time in this case is defined as the time that passes from the moment the Operator receives the scaling request to the moment the last microservice of the group is scaled.
The output times are stored in scaling_csv.txt


## 7. WIP

For now what I am trying to implement is a way of measuring times for scaling actions

Fix a bug that doesn't let me terminate the deployments and sometimes the Operator itself because some finalizers are not removed.
Everytime one wants to un-deploy an application, it should do a patch on the deployment and remove the finalizers. This is not a good practice and should be fixed. The following code is an example for TeaStore

```bash
kubectl patch deployments teastore-auth teastore-db teastore-image teastore-persistence teastore-image teastore-webui teastore-recommender teastore-registry -p '{"metadata": {"finalizers": []}}' --type merge
```





