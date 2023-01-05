import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.append(str(ROOT/"yolov7"))

from threading import Thread
from time import time

import torch
import torch.backends.cudnn as cudnn
from aiortc import VideoStreamTrack as _VideoStreamTrack
from av import VideoFrame
from models.experimental import attempt_load
from numpy import random
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

from .classify import classify, load_model

WEIGHTS = ROOT/"weights"

YOLO_WEIGHT = WEIGHTS/"best.pt"
MANGO_WEIGHT = WEIGHTS/"weight_mango_efficient.pth"
DRAGON_WEIGHT = WEIGHTS/"weight_dragon_efficient.pth"


class FruitTrackingModel:
    _instances: dict[str, "FruitTrackingModel"] = {}

    @classmethod
    def start(cls, url: str):
        if url not in cls._instances:
            cls._instances[url] = cls(url)
        return cls._instances[url]

    @classmethod
    def stop(cls, url: str):
        if url in cls._instances:
            cls._instances[url].running = False
            del cls._instances[url]

    def __init__(self, url: str):
        self.url = url

        self.current_det = None
        self.current_frame = None
        self.classify_result = []

        self.running = True

        detect_thread = Thread(target=self.detect)
        detect_thread.daemon = True
        detect_thread.start()
        self.detect_thread = detect_thread


        classify_thread = Thread(target=self.classify)
        classify_thread.daemon = True
        classify_thread.start()
        self.classify_thread = classify_thread

    def detect(self):
        while self.running:
            imgsz = 640
            conf_thres = 0.25
            iou_thres = 0.45
            weights = ROOT/"weights"/"best.pt"
            source = self.url
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

            device = select_device()
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size

            if half:
                model.half()  # to FP16

            # Set Dataloader
            if webcam:
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            # bgr
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            # detect
            colors = [
                [0, 0, 0],
                [0, 0, 0],
                [255, 0, 0], # blue
                [191, 64, 191], # purple
            ]
            # classify
            colors2 = [
                [0, 255, 0], # green
                [0, 255, 255], # yellow
                [0, 0, 255], # red
            ]
            # classes = [names.index(name) for name in names if name.startswith("pile")]
            classes = None

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1

            for path, img, im0s, vid_cap in dataset:
                if not self.running:
                    break
                # t1 = time()
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img)[0]

                # Inference
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    pred = model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes)

                # Process detections
                det = pred[0]
                if webcam:  # batch_size >= 1
                    im0 =im0s[0].copy()
                else:
                    im0 = im0s
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    self.current_det = det
                    # Add bbox to image
                    for *xyxy, conf, cls in reversed(det):
                        if cls > 1:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # label = None
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    for xyxy, cls in self.classify_result:
                        plot_one_box(xyxy, im0, label=None, color=colors2[int(cls)], line_thickness=2)

                self.current_frame = im0
                # t2 = time()
                # print(f"Detect time: {t2-t1}")

    def classify(self):
        models = [load_model(MANGO_WEIGHT), load_model(DRAGON_WEIGHT)]
        while self.running:
            t1 = time()
            if self.current_det is None or self.current_frame is None:
                continue
            det = self.current_det
            frame = self.current_frame
            result = []
            for *xyxy, conf, cls in reversed(det):
                if cls < 2:
                    im = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    cls = classify(im, models[int(cls)])
                    result.append((xyxy, cls))
            self.classify_result = result
            t2 = time()
            print(f"Classify time: {t2-t1}")

    def get_stream_track(self):
        return VideoStreamTrack(self)


class VideoStreamTrack(_VideoStreamTrack):

    def __init__(self, model):
        super().__init__()
        self.model = model

    async def recv(self):
        frame = self.model.current_frame
        if frame is None:
            return None
        frame = VideoFrame.from_ndarray(frame, format="bgr24")
        frame.pts, frame.time_base = await self.next_timestamp()
        return frame
