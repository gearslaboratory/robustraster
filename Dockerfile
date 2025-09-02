# syntax=docker/dockerfile:1.7
FROM daskdev/dask:2024.8.1-py3.10
#USER root

# Cache conda pkgs across layers for fast rebuilds
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install -n base -y -c conda-forge mamba \
 && mamba install -n base -y -c adrianom -c conda-forge --strict-channel-priority robustraster==0.4.1 \
 && conda clean -afy

#USER dask
