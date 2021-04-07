# Object Detection With MMDetection Framework and DeepMux

[*MMDetection*](https://github.com/open-mmlab/mmdetection) is a toolbox for object detection based on PyTorch. In this article we will show how to deploy a pretrained [Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) model with Deepmux.

## Setting up the environment.

To run everything, you need to install *deepmux-cli* and log in. This is done with the following two commands

```
pip install deepmux-cli
deepmux login 
```

The last command will ask you to enter your unique API token, which you can find on app.deepmux.com.

Let's clone the mmdetection repository:

`git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection`

## Downloading pretrained weights

Download the [checkpoint](http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) and put it into `./checkpoints` folder. This is one of the many models avaliable in mmdetection's [model zoo](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md).

## Writing some code:

The code we write here is based on (get_started.md)[https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md] provided in the original repo. We take an image as an input and need to return the results of object detection. In this example we will return it as an UTF-encoded string.

```python
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
```

## Adding requirements and initializing:

Since we are going to use a dedicated environment with all of the mmdetection's dependencies installed, we do not need to create a `requirements.txt` file!

So, we write the following command:

`deepmux init`

The deepmux.yaml file will appear in the project folder. Fill it in with your data:

```yaml
name: mmdet_segentation # project name
env: <...>
python:
  call: func:process # file and function to call
```

To fill in the env line, ru the deepmux env command. The result will be something like this:
```name: python3.6 language: python
name: python3.7 language: python
name: python3.6-tensorflow2.1-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow2.1-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow2.2-pytorch-1.6-cuda10.1 language: python
name: python3.7-tensorflow1.13.1-pytorch-1.3-cuda10.0 language: python
name: python3.7-mmdetection-pytorch-1.6-cuda10.1 language: python
```
We need the last one. The resulting yaml file should look like this:
```yaml
name: mmdet_segentation
env: python3.7-mmdetection-pytorch-1.6-cuda10.1
python:
  call: func:process
```
## Loading the model:

Your model is ready to deploy! Now, just call the `deepmux upload` command. This might take a few minutes to process.

## Running the model:

The model is uploaded and ready to use. Let's run the model on an image!

To do this, you need to run the following command:

```shell
curl -X POST -H "X-Token: <YOUR TOKEN>" https://api.deepmux.com/v1/function/mmdet_segentation/run --data-binary "@city_landscape.png" > result
```
and model's output will be saved to `result`, from which it can be parsed as a regular JSON.
