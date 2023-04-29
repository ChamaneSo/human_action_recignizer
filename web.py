import torch
import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="./")

@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class MyVideoCapture:
    
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        
    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img
    
    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)
        
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip
    
    def release(self):
        self.cap.release()
        
def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map, output_video ="", vis=False):
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box,im,color,text)
        im = im.astype(np.uint8)
        #output_video.write(im)
        im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        # if vis:
        #     cv2.imshow("demo", im)
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        return im
    
@app.get("/video_feed")
async def video_feed():

    # define a function to generate the frames with the predicted action
    def generate_frames():
        while True:
            
            device = "cuda"
            imsize = 640
            
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l6').to(device)
            model.conf = 0.4
            model.iou = 0.4
            model.max_det = 100
            
            
            video_model = slowfast_r50_detection(True).eval().to(device)
            
            deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
            ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
            coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
            
            cap = MyVideoCapture("demo6.mp4")
            id_to_ava_labels = {}
            a=time.time()
            num = 0
            while not cap.end:
                ret, img = cap.read()
                if not ret:
                    continue
                yolo_preds=model([img], size=imsize)
                yolo_preds.files=["img.jpg"]
                
                deepsort_outputs=[]
                for j in range(len(yolo_preds.pred)):
                    temp=deepsort_update(deepsort_tracker,yolo_preds.pred[j].cpu(),yolo_preds.xywh[j][:,0:4].cpu(),yolo_preds.ims[j])
                    if len(temp)==0:
                        temp=np.ones((0,8))
                    deepsort_outputs.append(temp.astype(np.float32))
                    
                yolo_preds.pred=deepsort_outputs
                
                if len(cap.stack) == 25:
                    print(f"processing {cap.idx // 25}th second clips")
                    clip = cap.get_video_clip()
                    if yolo_preds.pred[0].shape[0]:
                        inputs, inp_boxes, _=ava_inference_transform(clip, yolo_preds.pred[0][:,0:4], crop_size=imsize)
                        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
                        if isinstance(inputs, list):
                            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                        else:
                            inputs = inputs.unsqueeze(0).to(device)
                        with torch.no_grad():
                            slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                            slowfaster_preds = slowfaster_preds.cpu()
                        for tid,avalabel in zip(yolo_preds.pred[0][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
                            id_to_ava_labels[tid] = ava_labelnames[avalabel+1]
                
                        
                frame = save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map, "outputvideo", False)
                # encode the frame as JPEG
                _, buffer = cv2.imencode(".jpg", frame)

                # yield the encoded frame
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    # return the streaming response with the frames
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace;boundary=frame")
