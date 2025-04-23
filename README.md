# offgridplanner simulation-server

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Project page:
TODO

Developed by [Reiner Lemoine Institut](https://reiner-lemoine-institut.de/en/).

The offgridplanner simulation-server runs energy model optimization on demand and returns optimal solution, if any. It also runs optimization of building connection to a grid, based on pole and cable prices. An example architecture in which this simulation-sever is provided on the following picture

TODO new picture

The offgridplanner frontend sends simulation requests to a simulation server instance so that you can queue tasks and still use the app while the simulation to be done. The server can be running on your local computer, or you can set it up online.



The code in this repository creates the simulation server and a basic server API to dispatch simulation tasks to a queue of workers.
The API typically receives a post request with a json input file, sends this file to a parser which
initiate simulation. Once the simulation is done, a json response is sent back. The json results can also be retrieved with the task id.

## License

This project is licensed under `GNU AFFERO GENERAL PUBLIC LICENSE. See the LICENSE file for details.

## Prerequisite

You need to be able to run `docker-compose` commands, the simpler might be to install [docker desktop](https://www.docker.com/products/docker-desktop/)

## Get started

Run `sudo docker-compose up -d --build` to run the task queue and the webapp simultaneously.

Now the webapp is available at `127.0.0.1:5001`

Use

    sudo docker-compose logs web

to get the logs messages of the `web` service of the docker-compose.yml file


Run `sudo docker-compose down` to shut the services down.


### local deploy of the server

Once you ran the docker-compose command from [Get started menu](#Get started) above,
you should be able to visit http://127.0.0.1:5001 and see a page where you can upload json files to start a simulation. 
The `SIM_HOST_API` environment variable in open-plan GUI should then be set as `SIM_HOST_API=http://127.0.0.1:5001`.

### online deploy of the server

You need first to have access to online services to host the server (eg. one of those listed in https://geekflare.com/docker-hosting-platforms/). 
You might need to adapt the docker-compose.yml file to be able to access the docker container on a subdomain of your service provider. 
You can then visit a URL to see the page equivalent to http://127.0.0.1:5001 in [above section](#local deploy of the server). 
You need to link your open-plan gui to this URL.


## Develop while services are running

### Using [redis](https://redis.io/documentation)

#### Ubuntu [install instructions](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04)

    sudo apt update
    sudo apt install redis-server

Then go in redis conf file

    sudo nano /etc/redis/redis.conf

and look for `supervised` parameter, set it to `systemd`

    supervised systemd


Then start the service with

    sudo systemctl restart redis.service

or

    sudo service redis-server start

(to stop it use `sudo service redis-server stop`)
Move to `task_queue` and run `. setup_redis.sh` to start the celery queue with redis a message
 broker.

### Using [RabbitMQ](https://www.rabbitmq.com/getstarted.html)

### Using [fastapi](https://fastapi.tiangolo.com/)

In another terminal go the the root of the repo and run `. fastapi_run.sh`

Now the fastapi app is available at `127.0.0.1:5001`


## Docs

To build the docs simply go to the `docs` folder

    cd docs

Install the requirements

    pip install -r docs_requirements.txt

and run

    make html

The output will then be located in `docs/_build/html` and can be opened with your favorite browser

## Code linting

Use `black .` to lint the python files inside the repo

