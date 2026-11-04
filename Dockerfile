FROM python:3.10-slim-bookworm AS experimento_base
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG DGL_VERSION=2.1.0
ARG DGL_WHEELS=https://data.dgl.ai/wheels/repo.html
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DGLBACKEND=pytorch MPLBACKEND=Agg
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple pip==25.1.1
RUN flavor="${TORCH_INDEX##*/}" && python -m pip install --no-cache-dir --index-url https://pypi.org/simple "https://download-r2.pytorch.org/whl/${flavor}/torch-2.1.2%2B${flavor}-cp310-cp310-linux_x86_64.whl"
RUN python -m pip install --no-cache-dir -r requirements-docker.txt && python -m pip install --no-cache-dir "dgl==${DGL_VERSION}" -f ${DGL_WHEELS}

COPY BDSTGNN/ ./BDSTGNN/
COPY MepoGNN/ ./MepoGNN/
COPY comparar_activos.py datos_y_metricas.py ./

RUN python -m pip install --no-cache-dir "pydantic==2.10.6"
ENTRYPOINT ["python", "comparar_activos.py"]

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS bibliotecas_cuda

FROM experimento_base AS gpu
COPY --from=bibliotecas_cuda /usr/local/cuda-12.1/targets/x86_64-linux/lib/ /opt/cuda-12.1/lib/
RUN echo /opt/cuda-12.1/lib > /etc/ld.so.conf.d/pfc-cuda.conf && ldconfig && python -c "from ctypes import CDLL; CDLL('libcusparse.so.12')"

FROM experimento_base AS cpu
