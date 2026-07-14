ARG PYTHON_VERSION=3.11.11
ARG UV_VERSION=0.8.18

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-bookworm AS python-base
COPY --from=uv /uv /uvx /usr/local/bin/
ARG UV_PROJECT_ENVIRONMENT=/opt/python
ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:$PATH"

FROM python-base AS builder
WORKDIR /source
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --all-groups \
        --locked \
        --no-install-project

FROM python-base
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion \
        git \
        jq \
        less \
        locales \
        nano \
        vim \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen en_US.UTF-8

COPY --from=uv /uv /uvx /usr/local/bin/
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> /etc/bash_completion.d/uv

ARG APP_USER=metr
ARG APP_DIR=/home/${APP_USER}/app
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} ${APP_USER} \
 && useradd -m -u ${USER_ID} -g ${APP_USER} -s /bin/bash ${APP_USER} \
 && mkdir -p \
        /home/${APP_USER}/.config \
        ${APP_DIR} \
 && chown -R ${USER_ID}:${GROUP_ID} \
        /home/${APP_USER} \
        ${APP_DIR}

WORKDIR ${APP_DIR}

COPY --from=builder ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --all-groups --locked

CMD ["sleep", "infinity"]
