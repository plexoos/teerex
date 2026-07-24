# syntax=docker/dockerfile:latest

ARG OS=ubuntu24.04
ARG CUDA_VERSION=13.0.2

FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS} AS base

ARG TYPST_VERSION=0.14.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
 && apt install -y g++ gcc gzip tar python3 python-is-python3 python3-pip curl git \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt install -y nodejs \
 && npm install -g @openai/codex@0.145.0 \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
      "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-x86_64-unknown-linux-musl.tar.xz" \
      -o /tmp/typst.tar.xz \
 && tar -xJf /tmp/typst.tar.xz -C /tmp \
 && mv "/tmp/typst-x86_64-unknown-linux-musl/typst" /usr/local/bin/typst \
 && rm -rf /tmp/typst* \
 && typst --version

RUN apt update && apt install -y vim

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ENV PROJECT_HOME=/workspaces/teerex
ENV VIRTUAL_ENV=${PROJECT_HOME}/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

WORKDIR $PROJECT_HOME

# Install Python dependencies
COPY pyproject.toml uv.lock $PROJECT_HOME/
COPY teerex $PROJECT_HOME/teerex
RUN uv sync

ARG DEV_USER=dev

RUN if ! id -u "${DEV_USER}" >/dev/null 2>&1; then \
      useradd --create-home --user-group --shell /bin/bash "${DEV_USER}"; \
    fi

ENV HOME=/home/${DEV_USER}

USER ${DEV_USER}
