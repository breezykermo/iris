curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
docker build -t my-app docker/
docker run -it --name my-app-container my-app /bin/bash