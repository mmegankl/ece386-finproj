# Compatible with Jetpack 6.2
FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gpiod \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/ && apt clean

COPY requirements.txt .

RUN pip install --upgrade --no-cache-dir pip && \ 
    pip install --no-cache-dir -r requirements.txt
    
    
COPY ece386fin_proj.py .
ENV HF_HOME="/huggingface/"
# In Dockerfile for the GPIO setup
ENV JETSON_MODEL_NAME=JETSON_ORIN_NANO

ENTRYPOINT ["python", "ece386fin_proj.py"]




