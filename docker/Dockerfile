# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:24.04@sha256:a853f94d226358a79c740cfc7bce0c289748f3fe3488d921d038ccd752c61b60
FROM ${BASE_IMAGE} AS toolchain
ARG UBUNTU_SNAPSHOT=20261001T000000Z
ARG LLVM_PACKAGE_VERSION=1:21.1.8~++20251221032922+2078da43e25a-1~exp1~20251221153059.70
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
# Public ISRG X1/X2 and DigiCert G2 trust anchors bootstrap TLS; the signed
# ca-certificates package subsequently installs Ubuntu's complete trust store.
COPY docker/bootstrap-ca.crt /etc/ssl/certs/ca-certificates.crt
RUN sed -i "s|http://archive.ubuntu.com/ubuntu/|https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|; s|http://security.ubuntu.com/ubuntu/|https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|; s|http://ports.ubuntu.com/ubuntu-ports/|https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|" /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get -o Acquire::Check-Valid-Until=false -o APT::Update::Error-Mode=any update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg git openssh-client python3 python3-venv \
       gcc-13 g++-13 gfortran-13 make pkg-config libopenblas-dev libssl-dev zlib1g-dev libbz2-dev \
       libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libncurses-dev uuid-dev libgdbm-dev libexpat1-dev \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && printf '%s\n' 'deb [signed-by=/usr/share/keyrings/llvm.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-21 main' > /etc/apt/sources.list.d/llvm.list \
    && apt-get -o Acquire::Check-Valid-Until=false -o APT::Update::Error-Mode=any update \
    && apt-get install -y --no-install-recommends "clang-21=${LLVM_PACKAGE_VERSION}" "llvm-21=${LLVM_PACKAGE_VERSION}" "llvm-21-tools=${LLVM_PACKAGE_VERSION}"

FROM toolchain AS build-tools
WORKDIR /opt/evaluation
COPY build/ build/
COPY config/sources.json config/sources.lock.json config/toolchain.lock.json config/build-requirements.lock.txt config/

FROM build-tools AS sources
# All locked repositories and their submodules use public HTTPS URLs.
RUN python3 build/sources.py resolve \
    && python3 build/sources.py checkout --root /opt/crossjit/sources

FROM sources AS interpreters
ARG BUILD_JOBS=4
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --python-only

FROM interpreters AS calibration-build
ARG BUILD_JOBS=4
COPY scripts/common.py scripts/worker.py scripts/
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --numpy-only --calibration

FROM interpreters AS fixed-strategy-build
ARG BUILD_JOBS=4
ARG OPTIMIZATION_PRESET=v32
COPY config/optimizations/ config/optimizations/
COPY scripts/common.py scripts/worker.py scripts/
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --numpy-only --optimization-preset ${OPTIMIZATION_PRESET}

FROM interpreters AS production-build
ARG BUILD_JOBS=4
COPY scripts/common.py scripts/worker.py scripts/
COPY --from=policy / /opt/crossjit/policy/
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --numpy-only --policy /opt/crossjit/policy

FROM calibration-build AS calibration
RUN chmod a+r /opt/crossjit/build-manifest.json
COPY config/ config/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
ENTRYPOINT ["python3"]
CMD ["scripts/run_rq1_performance.py", "--help"]

FROM fixed-strategy-build AS fixed-strategy
RUN chmod a+r /opt/crossjit/build-manifest.json
COPY config/ config/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
ENTRYPOINT ["python3"]
CMD ["scripts/run_rq1_performance.py", "--help"]

FROM production-build AS production
RUN chmod a+r /opt/crossjit/build-manifest.json
COPY config/ config/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
ENTRYPOINT ["python3"]
CMD ["scripts/run_rq1_performance.py", "--help"]

# Validation using exact local mirrors; default sources stage always clones.
FROM build-tools AS offline-sources
COPY --from=source-cache / /opt/crossjit/sources/
RUN if [ -f /opt/crossjit/sources/source-lock.json ]; then \
      cp /opt/crossjit/sources/source-lock.json config/sources.lock.json; \
    fi \
    && python3 build/sources.py verify --root /opt/crossjit/sources

FROM offline-sources AS offline-interpreters
ARG BUILD_JOBS=4
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --python-only

FROM offline-interpreters AS calibration-offline-build
ARG BUILD_JOBS=4
COPY scripts/common.py scripts/worker.py scripts/
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --numpy-only --calibration

FROM offline-interpreters AS fixed-strategy-offline-build
ARG BUILD_JOBS=4
ARG OPTIMIZATION_PRESET=v32
COPY config/optimizations/ config/optimizations/
COPY scripts/common.py scripts/worker.py scripts/
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --numpy-only --optimization-preset ${OPTIMIZATION_PRESET}

FROM offline-interpreters AS production-offline-build
ARG BUILD_JOBS=4
COPY scripts/common.py scripts/worker.py scripts/
COPY --from=policy / /opt/crossjit/policy/
RUN python3 build/build_system.py --jobs ${BUILD_JOBS} --numpy-only --policy /opt/crossjit/policy

FROM calibration-offline-build AS calibration-offline
RUN chmod a+r /opt/crossjit/build-manifest.json
COPY config/ config/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
ENTRYPOINT ["python3"]
CMD ["scripts/run_rq1_performance.py", "--help"]

FROM fixed-strategy-offline-build AS fixed-strategy-offline
RUN chmod a+r /opt/crossjit/build-manifest.json
COPY config/ config/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
ENTRYPOINT ["python3"]
CMD ["scripts/run_rq1_performance.py", "--help"]

FROM production-offline-build AS production-offline
RUN chmod a+r /opt/crossjit/build-manifest.json
COPY config/ config/
COPY scripts/ scripts/
COPY docs/ docs/
COPY README.md .
ENTRYPOINT ["python3"]
CMD ["scripts/run_rq1_performance.py", "--help"]
