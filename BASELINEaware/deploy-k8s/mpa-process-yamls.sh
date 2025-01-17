#!/bin/bash

# Copyright 2018 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_ROOT=$(dirname ${BASH_SOURCE})/..

function print_help {
  echo "ERROR! Usage: mpa-process-yamls.sh <action> [<component>]"
  echo "<action> should be either 'create' or 'delete'."
  echo "<component> might be one of 'admission-controller', 'updater', 'recommender'."
  echo "If <component> is set, only the deployment of that component will be processed,"
  echo "otherwise all components and configs will be processed."
}

if [ $# -eq 0 ]; then
  print_help
  exit 1
fi

if [ $# -gt 2 ]; then
  print_help
  exit 1
fi

ACTION=$1
COMPONENTS="mpa-v1alpha1-crd-gen mpa-rbac updater-deployment recommender-deployment admission-controller-deployment"

if [ $# -gt 1 ]; then
  COMPONENTS="$2-deployment"
fi

for i in $COMPONENTS; do
  if [ $i == admission-controller-deployment ] ; then
    if [ ${ACTION} == create ] ; then
      (bash ${SCRIPT_ROOT}/deploy-k8s/admission-controller/gencerts.sh || true)
    elif [ ${ACTION} == delete ] ; then
      (bash ${SCRIPT_ROOT}/deploy-k8s/admission-controller/rmcerts.sh || true)
      (bash ${SCRIPT_ROOT}/deploy-k8s/admission-controller/delete-webhook.sh || true)
    fi
  fi
  kubectl ${ACTION} -f ${SCRIPT_ROOT}/deploy-k8s/$i.yaml || true
done
