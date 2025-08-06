from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo11n.pt")
results = model("./bus.jpg")
im = Image.open("./bus.jpg")

# list of [x1, y1, x2, y2]
for result in results:
    # print(result)
    for box in result.boxes:
        # print(box.cls)
        print(result.names[int(box.cls)])
        print(box.xyxy.tolist())
        xyxy = box.xyxy.tolist()[0]

        im1 = im.crop((xyxy[0],xyxy[1],xyxy[2],xyxy[3]))
        im1.show()
