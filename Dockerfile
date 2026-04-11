FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------------------
# System deps for builds (numpy/scipy, lxml, TA-Lib) + unixODBC + MS ODBC
# --------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential curl wget git ca-certificates gnupg apt-transport-https \
        libxml2-dev libxslt1-dev zlib1g-dev \
        libjpeg-dev libpng-dev \
        libopenblas-dev liblapack-dev gfortran \
        unixodbc unixodbc-dev; \
    # Add Microsoft repo key (no apt-key) and point to Debian 12 (bookworm)
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
      | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg; \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
      > /etc/apt/sources.list.d/microsoft-prod.list; \
    apt-get update; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 msodbcsql18 mssql-tools18; \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Build and install TA-Lib C library (as you had)
# -------------------------------------------------
RUN set -ex \
 && curl -fsSL -o /tmp/ta-lib.tgz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
 && tar -xzf /tmp/ta-lib.tgz -C /tmp \
 && cd /tmp/ta-lib* \
 && ./configure --prefix=/usr \
 && make \
 && make install \
 && rm -rf /tmp/ta-lib* /tmp/ta-lib.tgz

WORKDIR /app

# --------------------------
# Python deps (layer cache)
# --------------------------
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# PyTorch CPU
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# NLTK data
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p "$NLTK_DATA" && \
    python -c "import nltk; nltk.download('vader_lexicon', download_dir='$NLTK_DATA')"

# App code
COPY . /app

# Gunicorn entrypoint (Hugging Face sets $PORT)
CMD ["bash", "-lc", "gunicorn -w 1 -k gthread -b 0.0.0.0:${PORT:-7860} pytrade:app --timeout 180"]

