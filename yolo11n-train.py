from numpy import half
from torch import classes, device
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

model = YOLO("yolo11n.pt")
# dataset = 'coco.yaml'
dataset = 'coco120.yaml'

print("============================= yaml : yolo11n.pt ============================= ")
print(model.yaml)
print("============================================================================= ")

# model = model.to("cpu")
model.eval()

model.export(format="torchscript", optimize=True, device='cpu')

model.save('yolo11n-96-in-dim.pt')
model.save('yolo11n-160-in-dim.pt')
model.save('yolo11n-320-in-dim.pt')

print("==============================================================================")

model96 = YOLO('yolo11n-96-in-dim.pt')

# def on_train_epoch_end_96(arg: DetectionTrainer):
#     model96.save(f'yolo11n-96-in-dim-e{arg.epoch}.pt')

# model96.add_callback("on_train_epoch_end", on_train_epoch_end_96)

resultsTrain96 = model96.train(save=True, data=dataset, epochs=1, imgsz=96, device='cpu', batch=16, degrees=45.0, flipud=0.2, erasing=0.15, mixup=0.01, dropout=0.04, bgr=0.33, shear=25.0, perspective=0.009)
# print(resultsTrain96)
# resultsVal96 = model.val(data=dataset)
# print(resultsVal96)

model96 = model96.to("cpu")
model96.eval()
model96.save('yolo11n-96-in-dim.pt')
print(">>>>>>>>>>>>>>>>>>>> EXPORTING TRAINED MODEL : START <<<<<<<<<<<<<<<<<<<<<<<<<")
model96.export(format="torchscript", optimize=True, device='cpu', imgsz=96)
print(">>>>>>>>>>>>>>>>>>>> EXPORTING TRAINED MODEL : DONE  <<<<<<<<<<<<<<<<<<<<<<<<<")
# model96.export(format="torchscript", optimize=True, device='cpu', imgsz=96, classes=[0]) # person only

print("==============================================================================")

model160 = YOLO('yolo11n-160-in-dim.pt')

# def on_train_epoch_end_160(arg: DetectionTrainer):
#     model160.save(f'yolo11n-160-in-dim-e{arg.epoch}.pt')

# model160.add_callback("on_train_epoch_end", on_train_epoch_end_160)

resultsTrain160 = model160.train(save=True, data=dataset, epochs=1, imgsz=160, device='cpu', batch=16, degrees=45.0, flipud=0.2, erasing=0.15, mixup=0.02, dropout=0.04, bgr=0.33, shear=25.0, perspective=0.009)
# print(resultsTrain160)
# resultsVal160 = model.val(data=dataset)
# print(resultsVal160)

model160 = model160.to("cpu")
model160.eval()
model160.save('yolo11n-160-in-dim.pt')
print(">>>>>>>>>>>>>>>>>>>> EXPORTING TRAINED MODEL : START <<<<<<<<<<<<<<<<<<<<<<<<<")
model160.export(format="torchscript", optimize=True, device='cpu', imgsz=160)
print(">>>>>>>>>>>>>>>>>>>> EXPORTING TRAINED MODEL : DONE  <<<<<<<<<<<<<<<<<<<<<<<<<")
# model160.export(format="torchscript", optimize=True, device='cpu', imgsz=160, classes=[0]) # person only

print("==============================================================================")

model320 = YOLO('yolo11n-320-in-dim.pt')

# def on_train_epoch_end_320(arg: DetectionTrainer):
#     model320.save(f'yolo11n-320-in-dim-e{arg.epoch}.pt')

# model320.add_callback("on_train_epoch_end", on_train_epoch_end_320)

resultsTrain320 = model320.train(save=True, data=dataset, epochs=1, imgsz=320, device='cpu', batch=16, degrees=45.0, flipud=0.2, erasing=0.15, mixup=0.02, dropout=0.04, bgr=0.33, shear=25.0, perspective=0.009)
# print(resultsTrain320)
# resultsVal320 = model.val(data=dataset)
# print(resultsVal320)

model320 = model320.to("cpu")
model320.eval()
model320.save('yolo11n-320-in-dim.pt')
print(">>>>>>>>>>>>>>>>>>>> EXPORTING TRAINED MODEL : START <<<<<<<<<<<<<<<<<<<<<<<<<")
model320.export(format="torchscript", optimize=True, device='cpu', imgsz=320)
print(">>>>>>>>>>>>>>>>>>>> EXPORTING TRAINED MODEL : DONE  <<<<<<<<<<<<<<<<<<<<<<<<<")
# model320.export(format="torchscript", optimize=True, device='cpu', imgsz=320, classes=[0]) # person only

print("==============================================================================")