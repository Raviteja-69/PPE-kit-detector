# # Clone the YOLOv7 repository
# !git clone https://github.com/WongKinYiu/yolov7.git
# cd yolov7
#
# # Install dependencies
# pip install -U -r requirements.txt
#
# # Download YOLOv7 segmentation model weights
# python download.py --weights yolov7-tiny.pt
#
# # Example script to perform segmentation using OpenCV
# python detect_opencv.py --weights yolov7-tiny.pt --img-size 640 --source data/images/
#
# # For video input



# python detect_opencv.py --weights yolov7-tiny.pt --img-size 640 --source data/video.mp4

# For webcam input
# python detect_opencv.py --weights yolov7-tiny.pt --img-size 640 --source 0




import cv2
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_sync

# Load YOLOv7 model
weights = 'Yolo-weights/yolov7-seg.pt'  # Change this to the desired weights file
device = select_device('')
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())

# Load image or video file
source = '.../images/2.jpg'  # Change this to the path of your image, video, or webcam

img_size = 640
save_img = True

# Set up OpenCV capture
cap = cv2.VideoCapture(source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img0 = frame.copy()
    img = cv2.resize(img0, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img[np.newaxis] / 255.0
    img = torch.from_numpy(img).to(device)

    # Inference
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, merge=False, classes=None, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

    # Display the result
    cv2.imshow('YOLOv7 Segmentation', img0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
