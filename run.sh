# Install docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
# Build and run docker image from docker/Dockerfile
sudo docker build -t my-app docker/
sudo docker run -it --name my-app-container my-app /bin/bash