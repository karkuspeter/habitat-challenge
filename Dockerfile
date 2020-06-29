# FROM tensorflow/tensorflow:1.14.0-gpu-py3
FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

RUN /bin/bash -c ". activate habitat; pip install ipdb zmq flask networkx"

RUN /bin/bash -c ". activate habitat; conda install tensorflow-gpu=1.14.0"
#RUN /bin/bash -c ". activate habitat; pip install tensorflow-gpu==1.14 tensorpack configargparse socketIO-client"
RUN /bin/bash -c ". activate habitat; pip install tensorpack configargparse socketIO-client"

# RUN source activate habitat
#RUN pip install ipdb zmq flask networkx
#RUN pip install
RUN mkdir /temp
RUN mkdir /configs

ADD agent.py agent.py
ADD submission.sh submission.sh
ADD configs/challenge_pointnav2020.local.rgbd.yaml /challenge_pointnav2020.local.rgbd.yaml

ENV AGENT_EVALUATION_TYPE remote
ENV TRACK_CONFIG_FILE "/challenge_pointnav2020.local.rgbd.yaml"
ENV PYTHONPATH /mclnet:$PYTHONPATH
ENV PYTHONPATH /evalai-remote-evaluation:$PYTHONPATH
ENV CHALLENGE_CONFIG_FILE $TRACK_CONFIG_FILE
# ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

COPY *.py /
COPY *.sh /
COPY *.conf /
COPY data/habitat /data/habitat
COPY build/mclnet /mclnet

# For local eval only
COPY configs/* /configs/
COPY build/habitat-api /habitat-api
COPY build/habitat-sim /habitat-sim

# COPY cuda-10.0 /usr/local/cuda-10.0

# COPY build/benchmark.py /habitat-api/habitat/core/benchmark.py

### This is to update reset
#COPY ./GibsonEnvV2Clean/gibson2/envs/challenge.py /opt/GibsonEnvV2/gibson2/envs/challenge.py

WORKDIR /

CMD ["/bin/bash", "-c", "source activate habitat && bash submission.sh"]
