from ultralytics import YOLO

# Load a model
model = YOLO(r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\models\Disabled_640S_Best.pt')  # load the nano model

# Test on an image
results = model(r'C:\cornerstonex\machine_learning\Mobility-Aids-Detection\dataset\images\test\02_jpeg.rf.0352c7dbff7a331dd8ea3f828c8ef62e.jpg')