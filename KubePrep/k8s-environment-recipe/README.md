# K8S Environment Recipe

This repository contains [Kameleon](https://kameleon.imag.fr) recipes for generating a Kubernetes environment on top of Grid'5000 environments.

By executing these recipes, Kameleon first retrieves the last specified Grid'5000 environment (by default `debian11-min`), then installs all Kubernetes dependencies (*containerd*, *kubeadm*, ...) and finally exports the environment with all required files to be registered in the Grid'5000 Kadeploy registry.

## Installation

Kameleon is already installed on Grid'5000 nodes. Therefore, you can build this environment directly on Grid'5000 nodes without installing any dependencies.

If you want to build this Kubernetes environment on a different platform (e.g., a personal computer or a VM), you will need to install Kameleon as indicated in its own [documentation](https://kameleon.imag.fr/installation.html).

## Usage

In the following, we assume that the bash commands are either executed on a Grid'5000 frontend or a Grid'5000 node.

### Building the K8S Environment

```bash
node$ kameleon build debian11-k8s.yaml
```

#### Specifying *containerd* or *kubeadm* versions

You can specify *containerd* and *kubeadm* versions by either editing `debian11-k8s.yaml` (see `CONTAINERD_VERSION` or `KUBEADM_VERSION`) or specifying them directly in the Kameleon command:

```bash
node$ kameleon build debian11-k8s.yaml --global CONTAINERD_VERSION:1.6.24-1 KUBEADM_VERSION:1.28
```

The recipes instruct Kameleon to install *containerd* using the `containerd.io` deb package distributed by Docker and *kubeadm* using dkgs.k8s.io package repositories. Before specifying a custom version, please verify that the version is available on these deb repositories.

#### Giving a Version to the K8S Environment (used by Kadeploy)

By default, the version specified in the description environment (see below) is the current date: YYYYMMDD.

However, if you plan to build multiple K8S environments (e.g., to test different *containerd* versions or to upgrade *kubernetes* dependencies), you should probably define a more meaningful version to better identify what is included in the K8S environment.

You can specify the version by either editing `debian11-k8s.yaml` (see `grid5000_environment_export_version`) or specifying it directly in the Kameleon command:

```bash
node$ kameleon build debian11-k8s.yaml --global grid5000_environment_export_version:1111
```

Note: due to some constraints in Kadeploy, the version **MUST** be an integer.

### Registering the K8S Environment in Kadeploy Registry

Kameleon produces multiple files in your public repository:

* `debian11-k8s.dsc`: a description environment for [Kadeploy]. This file is used to register your environment in the Kadeploy registry.
* `debian11-k8s-[VERSION].tgz`: a tarball containing all the OS files included *containerd* and *Kubernetes* dependancies.
* `debian11-k8s-disable-swap.tar.gz`: a custom post-install to deactivate the swap partition on deployed nodes (required by Kubernetes).

To install the environment inside the Kadeploy registry, run the following command:

```bash
frontend$ kaenv -a ~/public/debian11-k8s
```

## Deploy Kubernetes on Grid'5000

Reserve two nodes (considering one node for the control plane and one node as a worker node):

```bash
frontend$ oarsub -I -t deploy -l nodes=2
```

#### Deploy the K8S Environment

Use Kadeploy to install the K8s Environnement on the reserved nodes.

```bash
frontend$ kadeploy3 debian11-k8s
```

#### Initialize the control-plane node

We consider in the following that the first node is the control-plane node. We use `kubeadm` to initialize it.

```bash
frontend$ ssh $(cat $OAR_NODEFILE | uniq | head -n 1) -l root
control-plane$ kubeadm init
```

(make a record of the `kubeadm join` command that `kubeadm init` outputs)

#### Install a networking add-on

We indicate bellow how to install *Flannel* or *Wease Net* add-on (choose one). See [this page](https://kubernetes.io/docs/concepts/cluster-administration/addons/#networking-and-network-policy) for a non-exhaustive list of networking addons supported by Kubernetes.

##### Flannel

[Flannel](https://github.com/flannel-io/flannel#deploying-flannel-manually) is a simple and easy way to configure a layer 3 network fabric designed for Kubernetes.

```bash
control-plane$ export KUBECONFIG=/etc/kubernetes/admin.conf
control-plane$ kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

##### Wease Net

[Weave Net](https://www.weave.works/docs/net/latest/kubernetes/kube-addon/#install) provides networking and network policy, will carry on working on both sides of a network partition, and does not require an external database.

```bash
control-plane$ export KUBECONFIG=/etc/kubernetes/admin.conf
control-plane$ https://github.com/weaveworks/weave/releases/download/v2.8.1/weave-daemonset-k8s.yamll
```

#### Join the worker node to the cluster

Use `kubeadm` to register a node as worker node.

```bash
frontend$ ssh $(cat $OAR_NODEFILE | uniq | tail -n 1) -l root
worker$ kubeadm join .....
```

(use the command output by `kubeadm init`)

## Troubleshooting

### Deploying the K8S Environment via Grid'5000 API:

By default, the path for the tarball (`debian11-k8s-[VERSION].tgz`) specified in the Kadeploy description environment (`debian11-k8s.dsc`) is not compatible for use through the Grid'5000 API. To make it compatible, you need to uncomment the following line in `debian11-k8s.yaml`:

```yaml
grid5000_environment_export_baseurl: "http://public.grenoble.grid5000.fr/~$USER/"
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0)
