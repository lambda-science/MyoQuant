FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /bin/uv

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/lambda-science/MyoQuant.git .
COPY myoquant /app/myoquant
WORKDIR /app/myoquant
RUN wget https://huggingface.co/corentinm7/MyoQuant-SDH-Model/resolve/main/model.h5
RUN uv sync

WORKDIR /app/myoquant/streamlit
ENTRYPOINT ["uv", "run", "streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0"]
