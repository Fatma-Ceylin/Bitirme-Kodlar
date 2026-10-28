from ultralytics import YOLO

model = YOLO("best.pt")
source = "C:\\Users\\fatma\\Desktop\\yolo-test\\input_video2.mp4"

# save=True ekledik
results = model(source, stream=True, save=True)

# stream=True olduğu için döngüyle kareleri ilerletmemiz şart:
for r in results:
    pass  # Model her kareyi işler ve otomatik kaydeder