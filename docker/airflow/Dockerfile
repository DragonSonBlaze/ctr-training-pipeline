# Airflow and the training job in one image, in two Python environments.
# Airflow runs from the official image's own Python. The project's pinned packages
# (MLflow, TensorFlow, PySpark) live in /opt/ctr-venv, so Airflow's pins and the
# project's pins never share an environment. The DAG calls the project through
# CTR_PYTHON=/opt/ctr-venv/bin/python.
FROM apache/airflow:3.4.0-python3.12

USER root

# Spark 4 in local mode needs a JVM. The base image's Debian repository offers Java 21. The JVM path depends on the CPU architecture, so a fixed symlink
# lets JAVA_HOME stay the same on arm64 and amd64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "/usr/lib/jvm/java-21-openjdk-$(dpkg --print-architecture)" /usr/lib/jvm/java-21
ENV JAVA_HOME=/usr/lib/jvm/java-21

# uv builds the project venv from the lock file. Copied from its official image, pinned to
# the version used on the Mac, because the Airflow image does not allow pip installs as root.
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

# Only the lock file is copied, so a code change does not rebuild this layer. The code
# itself is bind-mounted at /opt/ctr by docker-compose.yml.
COPY pyproject.toml uv.lock /tmp/ctr-deps/
WORKDIR /tmp/ctr-deps
RUN uv export --frozen --no-dev --no-hashes --no-emit-project --output-file requirements.txt \
    && uv venv /opt/ctr-venv --python 3.12 \
    && uv pip install --no-cache --python /opt/ctr-venv/bin/python -r requirements.txt \
    && rm -rf /tmp/ctr-deps

WORKDIR /opt/airflow
USER airflow
