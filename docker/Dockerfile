FROM mlebench-env:latest

ENV NONROOT_USER=nonroot
ENV NONROOT_HOME=/home/nonroot
ENV MLEVOLVE_ENV_NAME=mlevolve
ENV MLEVOLVE_PYTHON_VERSION=3.11
ENV MLEVOLVE_HOME=${NONROOT_HOME}/MLEvolve-lib

USER ${NONROOT_USER}

RUN micromamba create -y -n "${MLEVOLVE_ENV_NAME}" python="${MLEVOLVE_PYTHON_VERSION}" pip \
    && micromamba clean --all --yes

RUN mkdir -p ${MLEVOLVE_HOME}
WORKDIR ${MLEVOLVE_HOME}

COPY --chown=${NONROOT_USER}:${NONROOT_USER} requirements_base.txt ${MLEVOLVE_HOME}/requirements_base.txt
COPY --chown=${NONROOT_USER}:${NONROOT_USER} requirements_ml.txt ${MLEVOLVE_HOME}/requirements_ml.txt
COPY --chown=${NONROOT_USER}:${NONROOT_USER} requirements_domain.txt ${MLEVOLVE_HOME}/requirements_domain.txt

RUN micromamba run -n "${MLEVOLVE_ENV_NAME}" pip install --no-cache-dir --no-deps -r requirements_base.txt
RUN micromamba run -n "${MLEVOLVE_ENV_NAME}" pip install --no-cache-dir --no-deps -r requirements_ml.txt
# RUN micromamba run -n "${MLEVOLVE_ENV_NAME}" pip install --no-cache-dir --no-deps -r requirements_domain.txt \
RUN micromamba clean --all --yes

# entrypoint in mlebench-env:latest requires root
USER root
