import torch

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Export the model to ONNX
onnx_path = "yolov5s.onnx"
model.eval()
dummy_input = torch.randn(1, 3, 640, 640)  # Example input size
torch.onnx.export(model, dummy_input, onnx_path, opset_version=12)
print(f"Model exported to {onnx_path}")