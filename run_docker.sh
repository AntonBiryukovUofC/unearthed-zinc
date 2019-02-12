#!/bin/sh 

docker build . -t unearthed_zinc
#docker run -it -u $(id -u):$(id -g) -v /home/mjbirdge/Development/Python/unearthed-zinc:/unearthed-zinc unearthed_zinc
docker run -it -v /home/mjbirdge/Development/Python/unearthed-zinc:/unearthed-zinc unearthed_zinc
