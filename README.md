# Build image
docker build -t minerl-vpt .

# Chạy container với volume và GPU
docker run -it --rm \
  --gpus all \
  -v D:/Git/hoanganhle225/aiagent/aiagent-python:/workspace \
  minerl-vpt

