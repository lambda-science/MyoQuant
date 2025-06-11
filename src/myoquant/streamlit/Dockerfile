FROM python:3.12-slim-bullseye
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /bin/uv

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1
ENV PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
WORKDIR /app
RUN chown -R user:user /app
RUN chown -R user:user /home/user

COPY README.md /app/
COPY pyproject.toml /app/
COPY pages /app/pages
COPY run.py /app/run.py
RUN uv sync

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["uv", "run", "streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0"]