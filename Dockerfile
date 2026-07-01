FROM waggle/plugin-base:1.1.1-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8"

WORKDIR /src

RUN curl https://s3-west.nrp-nautilus.io/smokeynet/model.onnx -o /src/model.onnx

COPY src/requirements.txt requirements.txt
RUN pip3 install --upgrade -r /src/requirements.txt

COPY src /src

ENTRYPOINT [ "python3","/src/main.py" ]