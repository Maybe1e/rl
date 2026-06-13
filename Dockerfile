# Docker image for RL Robotic Arm Drawing
# Build: docker build -t rl-arm-drawing .
# Run:   docker run --gpus all -v $(pwd)/results:/app/results rl-arm-drawing

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rl_project/ ./rl_project/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1

CMD ["python3", "rl_project/main.py"]
