FROM python:3.12

WORKDIR /app

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

# Install system dependencies required by Sonic Annotator
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    unzip \
    libsndfile1 \
    libraptor2-0 \
    libserd-0-0 \
    liblilv-0-0 \
    libsamplerate0 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the Linux binaries for Sonic Annotator
RUN wget https://github.com/sonic-visualiser/sonic-annotator/releases/download/sonic-annotator-1.7/sonic-annotator-1.7.0-linux64-static.tar.gz \
    && tar -xzf sonic-annotator-1.7.0-linux64-static.tar.gz \
    && rm sonic-annotator-1.7.0-linux64-static.tar.gz \
    # Move the executable to a directory in the PATH
    && mv sonic-annotator*/sonic-annotator /usr/local/bin/sonic-annotator

# Example: Add a step to install a specific plugin after system dependencies
RUN wget https://github.com/bbc/bbc-vamp-plugins/releases/download/v1.1/Linux.64-bit.tar.gz \
    && tar -xzf Linux.64-bit.tar.gz \
    && mkdir -p /usr/local/lib/vamp/ \
    && mv bbc-vamp-plugins.so /usr/local/lib/vamp/ \
    && rm Linux.64-bit.tar.gz

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
    
    