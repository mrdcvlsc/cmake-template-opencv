from ultralytics import YOLO

models = [
    { 'name': 'yolo11n.pt', 'input': 640 }, 
    { 'name': 'yolo11n-96-in-dim.pt', 'input': 96 }, 
    { 'name': 'yolo11n-160-in-dim.pt', 'input': 160 }, 
    { 'name': 'yolo11n-320-in-dim.pt', 'input': 320 },
]

yolo_model = YOLO('yolo11n.pt')
yolo_model.save(models[1]['name'])
yolo_model.save(models[2]['name'])
yolo_model.save(models[3]['name'])

def convert_to_torchscript(model_info):
    model = YOLO(model_info['name'])
    print('model name :', model.model_name)
    model = model.to("cpu")
    model.eval()
    model.export(format="torchscript", optimize=True, device='cpu', imgsz=model_info['input'])


for model in models:
    print('====================================================')
    print(model)
    convert_to_torchscript(model)
    print('====================================================')
