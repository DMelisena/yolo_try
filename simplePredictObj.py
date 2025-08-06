from ultralytics import YOLO
import json

# Load a model
model = YOLO("yolo11n.pt")

# Predict with the model
results = model("./bus.jpg")

def test(results):
    array=[]
    # array.append()
    xyxyxy = results.boxes.clsresult.xyxy
    print("=====")
    print(xyxyxy)
    print("=====")
    for i in xyxyxy:
        array.append(i)
test(results)
# xyresult = test(results)
# print(xyresult)
# def format_results_as_objects(results):
#     output_data = []
#
#     for i, result in enumerate(results):
#         image_result = {
#             "image_id": i + 1,
#             "detections": []
#         }
#
#         if result.boxes:
#             for j, (cls, conf, box) in enumerate(zip(result.boxes.clsresult.boxes.xyxy)):
#                 class_name = result.names[int(cls)]
#                 confidence = conf.item()
#                 x1, y1, x2, y2 = box.tolist()
#
#                 detection = {
#                     "object_id": j + 1,
#                     "class": class_name,
#                     "confidence": round(confidence, 2),
#                     "bounding_box": {
#                         "x1": round(x1, 2),
#                         "y1": round(y1, 2),
#                         "x2": round(x2, 2),
#                         "y2": round(y2, 2)
#                     }
#                 }
#
#                 image_result["detections"].append(detection)
#
#         output_data.append(image_result)
#
#     return output_data
#
# # Get the structured object output
# detection_results = format_results_as_objects(results)
#
# # Save the objects as JSON
# with open("yolo_results.json", "w") as f:
#     json.dump(detection_results, f, indent=2)
#
# # Print the results as objects
# print("Detection Results:")
# for image_result in detection_results:
#     print(f"Image {image_result['image_id']}:")
#     if not image_result['detections']:
#         print("  No objects detected.")
#     else:
#         for detection in image_result['detections']:
#             print(f"  Object {detection['object_id']}:")
#             print(f"    Class: {detection['class']}")
#             print(f"    Confidence: {detection['confidence']}")
#             print(f"    Bounding Box: {detection['bounding_box']}")
#     print()
#
# print("Results have been saved to 'yolo_results.json'")
