# Builds docker image
sudo docker build $DOCKER_PROXY_BUILD_ARGS \
    -t ncf_keras:cpu_py3 -f Dockerfile_cpu_py3 .
