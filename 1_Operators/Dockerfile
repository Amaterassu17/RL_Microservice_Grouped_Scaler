FROM python:3.11-slim
ADD src/ /src
WORKDIR /src
ENV KUBERNETES_POD true
ENV GROUP_FILE ./groups.txt
ENV APP_NAME teastore
ENV NAMESPACE default
# COPY /src/requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install kopf kubernetes pyyaml pykube



EXPOSE 5000


# CMD ["kopf", "run", "/src/hooks.py", "-n", "default"]
#to debug so you can ssh
#we issue the following command
#command: [ "/bin/bash", "-c", "--" ]
# args: [ "while true; do sleep 30; done;" ]

# CMD ["/bin/bash", "-c", "while true; do sleep 30; done;"]
CMD ["kopf", "run", "hooks_horizontal.py", "-n", "default", "--dev", "--standalone"]
