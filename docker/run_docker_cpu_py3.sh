home_dir="$HOME"

# Runs docker image and attaching parent directory
sudo docker run $DOCKER_PROXY_RUN_ARGS \
-v $home_dir:$home_dir \
-p 8888:8888 \
-p 6006:6006 \
-it ncf_keras:cpu_py3 \
bash
