FROM debian:latest
MAINTAINER Juan F Paulini "jpaulini@gmail.com"
RUN apt-get -yqq update ;
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV PATH /opt/conda/bin:$PATH
ENV APP_NAME cv_app
ENV APP_HOME /opt/${APP_NAME}
ENV ENVIRONMENT dev 

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
 libglib2.0-0 libxext6 libsm6 libxrender1 \
  git mercurial subversion 

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

 RUN pip install --upgrade pip
 RUN pip install tensorflow
 RUN pip install matplotlib
 RUN pip install opencv-python
 RUN pip install pillow
 RUN pip install h5py
 RUN pip install keras
 RUN pip install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl 

RUN pip install redis
RUN pip install flask
RUN pip install flask_restful

RUN mkdir ${APP_HOME}
ADD cv_app ${APP_HOME}
RUN wget --quiet https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5 -O ${APP_HOME}/models/resnet50_coco_best_v2.0.1.h5

WORKDIR ${APP_HOME}

CMD [ "/bin/bash" ]