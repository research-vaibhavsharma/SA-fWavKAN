FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .

RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy C++ extensions and compile
COPY csrc/ ./csrc/
COPY setup.py .
RUN python3 setup.py install

COPY safwavkan/ ./safwavkan/
CMD ["python3", "-c", "import safwavkan; print('SA-fWavKAN environment ready.')"]
