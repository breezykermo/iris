curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo docker build -t my-app docker/
sudo docker run -it --name my-app-container my-app /bin/bash