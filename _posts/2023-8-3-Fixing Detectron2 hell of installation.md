---
layout: post
title: Fixing Detectron2 hell of installation
---


Created: July 25, 2023 2:34 PM
Created By: Mohit

It’s easy to train a Mask RCNN with detectron2, isn’t it? I haven’t seen a quicker solution to train a segmentation models and there’s barely any hassle. However, detectron2 has been a pain in production. First, because it seems to have strict installation requirements. Which I’m going to talk about here. Second, memory leaks if not served! Third, lack of flexibility to customise (quantize, prune, etc.) unlike Vanilla torch models. The off the shelf solutions for scripting or tracing and pushes you into an hell hole of bugs.

Today, I wanted to profile one of our detectron model that is in production. To test against a newly trained model. Unsurprisingly, I ran into installation issues. To be honest, in hindsight, I have made silly errors. Maybe have been ignorant. But can’t help but blame detectron2 because this is the case every damn time.

Let’s say I have the following pyproject.toml file

```bash
[tool.poetry]
name = "test"
version = "1.0.0"
description = "A repository for "
authors = ["Mohit Motwani <>"]

[tool.poetry.dependencies]
python = "^3.8"
ipython = "*"

[tool.poetry.group.test]
optional = true

[tool.poetry.group.test.dependencies]
numpy = "^1.24.1"
torch = "1.10.0"
torchvision = "0.11.1"
Pillow = "*"
opencv-python = "^4.7.0"
detectron2 = {url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/detectron2-0.6%2Bcu113-cp38-cp38-linux_x86_64.whl"}
memory_profiler = "*"
visdom = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

We download detectron 2 with requirements cuda 11.3, torch 1.10 and python3.8 (Look at the URL). I’m using cuda 11.7

I use poetry to setup my environment but don’t let that scare you. You can just write a requirements.txt to install with pip but you may have to be a lil more specific with your versions.

```bash
poetry install --with test
```

(Actually `poetry install` is enough as well)

All well and good.

I have the following started code:

```python
import argparse
import cv2
import torch
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_args():
    parser = argparse.ArgumentParser(description="Detectron2 profile")
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        required=True,
    )
    parser.add_argument(
        "--image", default="", metavar="FILE", help="path to image file", required=True
    )
    # parser.add_argument("--output", default="", metavar="FILE", help="path to output file")
    parser.add_argument(
        "--cpu",
        default=False,
        action="store_true",
        help="use CPU instead of GPU",
    )

    return parser.parse_args()

def profile_model(model, image, calls, sort_key, profiler_kwargs):
    with profile(**profiler_kwargs) as prof:
        with record_function("model_inference"):
            for i in range(calls):
                model(image)
                prof.step()

    print(prof.key_averages().table(sort_by=sort_key, row_limit=-1))

def predict(image, model):
    return model(image)

def load_model(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = (
        "models/model_final.pth"
    )
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    model = DefaultPredictor(cfg)
    return model

def main():
    args = get_args()
    print(args)
    model = load_model(args)
    image = cv2.imread(args.image)
    preds = predict(image, model)
    return preds

if __name__ == "__main__":
    output = main()
```

A fairly simple script. Here are the following steps

1. Parse the arguments
2. Load the model
3. load the image
4. predict the mask/objects/(anything) about the image

```python
python test.py --config_file config.yml --image image.jpg
```

Running the script gives the following error:

```python
module 'PIL.Image' has no attribute 'LINEAR'
```

Well, googling it I  find this GitHub issue:

[https://github.com/facebookresearch/detectron2/issues/5010](https://github.com/facebookresearch/detectron2/issues/5010)

They have changed the code from `LINEAR` to `BILINEAR`. 

that I could just go in the detectron code and change the `LINEAR` TO `BILINEAR`. That’s in file /`path/to/venv/site-packages/detectron2/data/transforms/transform.py` in the **init** of class ExtentTransform.

Or I could downgrade my Pillow version to `9.0.0` and reinstall with poetry. If you are using poetry, just the run the `poetry update` command

Okay, error one solved. Let’s rerun 

And we get an error:

```python
libtorch_cuda_cu.so: cannot open shared object file: No such file or directory
```

Now we are finally in the detectron2 installation hell hole

[https://github.com/facebookresearch/detectron2/issues/3614](https://github.com/facebookresearch/detectron2/issues/3614)

One users says degrading 11.0 to 10.1 solved the issue. This is not an ideal solution since I have other libraries depending on my current cuda. 

[https://github.com/facebookresearch/detectron2/issues/1365](https://github.com/facebookresearch/detectron2/issues/1365)

This required changing torch versions. But I have installed  the required torch version already.

The I tumble upon two links - one is the detectron2’s installation itself and the other is the almighty stack overflow 

[Installation — detectron2 0.6 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

This mentions running the following code and checking the cuda versions

```bash
python -m detectron2.utils.collect_env
```

The version of NVCC you use to build detectron2 or torchvision does not match the version of CUDA you are running with. This often happens when using anaconda's CUDA runtime.

Use `python -m detectron2.utils.collect_env` to find out inconsistent CUDA versions. In the output of this command, you should expect “Detectron2 CUDA Compiler”, “CUDA_HOME”, “PyTorch built with - CUDA” to contain cuda libraries of the same version.

When they are inconsistent, you need to either install a different build of PyTorch (or build by yourself) to match your local CUDA installation, or install a different version of CUDA to match PyTorch.

To my horror I notice this

```python
'1.10.0+cu102'
```

This mentions using installing the same torch but compiled with cuda 11.3. SMH

[OSError: libcudart.so.10.2: cannot open shared object file: No such file or directory](https://stackoverflow.com/questions/69934320/oserror-libcudart-so-10-2-cannot-open-shared-object-file-no-such-file-or-dire)

Let’s try making that change to our `pyproject.toml`

```python
[tool.poetry.group.test.dependencies]
numpy = "^1.24.1"
torch = { url = 'https://download.pytorch.org/whl/cu111/torch-1.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl' }
torchvision = { url = 'https://download.pytorch.org/whl/cu111/torchvision-0.11.2%2Bcu111-cp38-cp38-linux_x86_64.whl' }
Pillow = "9.0.0"
opencv-python = "^4.7.0"
detectron2 = { url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/detectron2-0.6%2Bcu113-cp38-cp38-linux_x86_64.whl" }
memory_profiler = "*"
visdom = "*"
```

or you could use

```python
poetry add torch@https://download.pytorch.org/whl/cu111/torch-1.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl
poetry add torchvision@https://download.pytorch.org/whl/cu111/torchvision-0.11.2%2Bcu111-cp38-cp38-linux_x86_64.whl
```

Now I have torch 1.10+cu111 and it works.

The main issue was the installed torch 1.10 was compiled with cuda 10.2 instead of cuda 11.3. But torch compiled with cuda 11.1 works because in the [official installation docs](https://pytorch.org/get-started/previous-versions/) , you don’t see 1.10.1 compiled with 11.3. But this works!

I spent 4 hours. It is a simple solution. I probably should have read the docs better but this was a frustrating experience regardless. I’m just glad that after numerous trials with docker and dev containers (as suggested by chatGPT) different version installations of detectron2, I was able to understand and solve the error. I hope you don’t have to waste as much time installing it.