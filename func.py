
from mmdet.apis import init_detector, inference_detector
import json


config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)

def process(img_bytes):
    with open('img_temp', 'wb') as f:
        f.write(img_bytes)

    gg = inference_detector(model, 'img_temp')

    return json.dumps([g.tolist() for g in gg])
                        
