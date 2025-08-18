from ultralytics import YOLO

model = YOLO("retuve_yolo_plugin/weights/hip-yolo-us.pt")

model.export(
    format="onnx",
    batch=1,
    optimize=True,
    device="cpu",
    simplify=False,
    nms=True,
    dynamic=False,
)
