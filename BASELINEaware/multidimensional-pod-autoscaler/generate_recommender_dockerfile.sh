#!/bin/bash

# Get the value of the CONTROL_IP environment variable
CONTROL_IP=$(kubectl get nodes -o wide | grep control-plane | awk '{print $6}')
PROMETHEUS_PORT=$(kubectl get services $PROMETHEUS_SERVICE -n monitoring -o wide | tail -n 1 |awk '{print $5}' | cut -d':' -f2 | cut -d'/' -f1)
# Generate the YAML code
generated=$(cat <<EOF
FROM gcr.io/distroless/static:latest
MAINTAINER Krzysztof Grygiel "kgrygiel@google.com"

ARG ARCH
COPY recommender-\$ARCH /recommender

ENTRYPOINT ["/recommender"]
CMD ["--v=4", "--stderrthreshold=info", "--prometheus-address=http://$CONTROL_IP:30090", "prometheus-cadvisor-job-name=kubernetes-nodes-cadvisor"]
EOF
)

output_path="./pkg/recommender/Dockerfile"

echo "$generated" > "$output_path"


