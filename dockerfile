# Use OpenFOAM development image v2312 (Ubuntu 22.04 based)
FROM opencfd/openfoam-dev:2312

ARG TARGETPLATFORM

USER root

# Install system dependencies (ensure python3-pip and python3-venv are included)
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    libssl-dev \
    flex \
    m4 \
    subversion \
    git \
    mercurial \
    wget \
    ca-certificates \
    bzip2 \
    sqlite3 \
    bear \
    curl \
    python3-pip \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure ofuser group/user (UID/GID 1000) exist and create home directory
RUN groupadd -g 1000 ofuser || echo "Group 1000 exists" && \
    useradd -u 1000 -g 1000 -ms /bin/bash -d /home/ofuser ofuser || echo "User 1000 exists" && \
    mkdir -p /home/ofuser

# Define OpenFOAM user project directory (still needed for OpenFOAM setup)
ENV OF_USER_PROJECT_DIR=/home/ofuser/OpenFOAM/ofuser-v2312
RUN mkdir -p ${OF_USER_PROJECT_DIR} && chown 1000:1000 ${OF_USER_PROJECT_DIR} # Ensure ownership

# --- Copy ALL necessary application files/dirs to /home/ofuser ---
COPY --chown=1000:1000 requirements.txt /home/ofuser/requirements.txt
COPY --chown=1000:1000 reactor_design_problem /home/ofuser/reactor_design_problem
COPY --chown=1000:1000 utils.py /home/ofuser/utils.py
COPY --chown=1000:1000 mesh_generation /home/ofuser/mesh_generation

COPY --chown=1000:1000 . /home/ofuser

# Ensure overall ownership for user home directory
RUN chown -R 1000:1000 /home/ofuser

# Switch to standard OpenFOAM user's UID
USER 1000
# Set WORKDIR to the project root
WORKDIR /home/ofuser

# Compile swak4Foam (ensure paths are correct relative to new WORKDIR if needed)
RUN mkdir /home/ofuser/tmp_build && \
    cd /home/ofuser/tmp_build && \
    bash -c "source /usr/lib/openfoam/openfoam2312/etc/bashrc && \
            echo 'Cloning swak4Foam...' && \
            hg clone http://hg.code.sf.net/p/openfoam-extend/swak4Foam swak4Foam && \
            cd swak4Foam && \
            echo 'Checking out develop branch...' && \
            hg update develop && \
            echo 'Compiling swak4Foam requirements (Bison)...' && \
            chmod +x ./maintainanceScripts/compileRequirements.sh ./Allwmake && \
            ./maintainanceScripts/compileRequirements.sh && \
            echo 'Compiling swak4Foam...' && \
            export WM_NCOMPPROCS=\$(nproc) && \
            ./Allwmake > \"\$HOME/log.swak4foam.make\" 2>&1 && \
            ./Allwmake >> \"\$HOME/log.swak4foam.make\" 2>&1 && \
            echo 'Moving compiled files...' && \
            mkdir -p \"\${FOAM_USER_LIBBIN}\" \"\${FOAM_USER_APPBIN}\" && \
            mv platforms/\"\${WM_OPTIONS}\"/lib/* \"\${FOAM_USER_LIBBIN}\"/ 2>/dev/null || true && \
            mv platforms/\"\${WM_OPTIONS}\"/bin/* \"\${FOAM_USER_APPBIN}\"/ 2>/dev/null || true" && \
    echo 'Cleaning up swak4Foam build directory...' && \
    cd /home/ofuser && rm -rf /home/ofuser/tmp_build

    RUN \
    # Ensure the install script uses the correct HOME directory
    HOME=/home/ofuser bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' && \
    # Create virtual environment using the full path to uv (now expected in /home/ofuser/.local/bin)
    /home/ofuser/.local/bin/uv venv /home/ofuser/venv --python=3.10 && \
    # Install packages from requirements.txt into the venv using the full path to uv
    /home/ofuser/.local/bin/uv pip install -r /home/ofuser/requirements.txt --python /home/ofuser/venv/bin/python && \
    # Clean cache using the full path to uv
    /home/ofuser/.local/bin/uv cache clean && \
    # Add venv activation to .bashrc
    echo ". /home/ofuser/venv/bin/activate" >> /home/ofuser/.bashrc

# Add uv's actual install path and the venv bin path to the final PATH
ENV PATH="/home/ofuser/.local/bin:/home/ofuser/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/home/ofuser/venv"

# Set Flask Environment Variables (Using module path is often more robust with PYTHONPATH)
ENV FLASK_APP=reactor_design_problem.functions:app
# ENV FLASK_APP=/home/ofuser/reactor_design_problem/functions.py # Alternative if module path fails
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Ensure final working directory is project root
WORKDIR /home/ofuser

# Default command: Source OpenFOAM env, run Flask (venv activated by .bashrc or implicitly by PATH)
CMD ["bash", "-c", "source /usr/lib/openfoam/openfoam2312/etc/bashrc && echo 'Starting Flask app...' && exec flask run"]