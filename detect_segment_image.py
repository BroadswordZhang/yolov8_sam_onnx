import cv2
import pathlib
import sys 
import numpy as np

sys.path.append(".")

from yolov8 import YOLOv8
from sam_onnx import SegmentAnythingONNX

# Initialize yolov8 object detector
detect_model_path = "models/yolov8m.onnx"
detector_model = YOLOv8(detect_model_path, conf_thres=0.2, iou_thres=0.3)

encoder_model = "models/sam_vit_h_4b8939.encoder.onnx"
decoder_model = "models/sam_vit_h_4b8939.decoder.onnx"
segment_model = SegmentAnythingONNX(
    encoder_model,
    decoder_model,
)
is_output = True
is_show = None 
# Read image
img_path = "images/plants.png"
output_path = "output/plants.png"
img = cv2.imread(img_path)

# Detect Objects
boxes, scores, class_ids = detector_model(img)

# create prompt [{'type': 'point', 'data': [575, 750], 'label': 0}, {'type': 'rectangle', 'data': [425, 600, 700, 875]}]
prompt, sub_prompt = [], {}
sub_prompt["type"] = "rectangle"
for box in boxes:
    sub_prompt["data"] = [int(i) for i in list(box)]
    prompt.append(sub_prompt.copy())
    


# prompt = json.load(open(args.prompt))
print(prompt)
# sam 
embedding = segment_model.encode(img)
masks = segment_model.predict_masks(embedding, prompt)
print(np.shape(masks))

# Save the masks as a single image.
mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
for m in masks[0, :, :, :]:
    mask[m > 0.0] = [0, 0, 255]
# Binding image and mask
visualized = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
cv2.imwrite("output/0.jpg", visualized)
# Draw the prompt points and rectangles.
i =1
for p in prompt:
    if p["type"] == "point":
        color = (
            (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
        )  # green for positive, red for negative
        cv2.circle(visualized, (p["data"][0], p["data"][1]), 10, color, -1)
    elif p["type"] == "rectangle":
        cv2.rectangle(
            visualized,
            (p["data"][0], p["data"][1]),
            (p["data"][2], p["data"][3]),
            (0, 255, 0),
            2,
        )
        cv2.imwrite("output/{i}.jpg".format(i=i), visualized)
    i +=1 
# 
if is_output is not None:
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, visualized)

if is_show:
    cv2.imshow("Result", visualized)
    cv2.waitKey(0)


# # Draw detections
# combined_img = yolov8_detector.draw_detections(img)
# # cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
# # cv2.imshow("Detected Objects", combined_img)
# cv2.imwrite("doc/img/273271,1b9330008da38cd6.jpg", combined_img)
# # cv2.waitKey(0)
