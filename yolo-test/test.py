from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# model = YOLO("yolo26n-pose.pt")  # yolo26n-pose.pt is a pretrained model for pose estimation yolo26n

# Single stream with batch-size 1 inference
source = 0



# Run inference on the source
results = model(source, stream=True)  # generator of Results objects