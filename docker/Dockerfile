FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

RUN pip install pytorch-pretrained-bert
RUN pip install emoji --upgrade
RUN pip install numpy==1.14.5
RUN pip install scikit-learn==0.19.1
RUN pip install scipy==1.0.1
RUN pip install snakeviz==1.0.0
RUN pip install tensorboard==1.10.0
RUN pip install tensorboardX==1.4

RUN apt-get update && apt-get install screen

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]
