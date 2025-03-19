from ultralytics import YOLO

model = YOLO("yolo11n.pt")

print("============================= yaml : yolo11n.pt ============================= ")
print(model.yaml)
print("============================================================================= ")

model = model.to("cpu")
model.eval()
model.export(format="torchscript")

results = model("https://ultralytics.com/images/bus.jpg")
print(results)

model.predict("https://ultralytics.com/images/bus.jpg", save=True, imgsz=320, conf=0.5)