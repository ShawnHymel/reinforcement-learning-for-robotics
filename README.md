# Reinforcement Learning for Robotics

## Installation

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Make sure it is running before continuing to the next step.

Open a terminal and build the Docker image:

```sh
docker build -t rl-robotics -f Dockerfile.cpu .
```

Run the image (note: VS Code is memory hungry, so we bump the shared memory up to 2 GB):

```sh
docker run -it --rm -p 3000:3000 -v "${PWD}/workspace:/workspace" --shm-size=2g rl-robotics
```
