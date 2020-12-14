# Triton Server inference issues with multiple models #

This repo contains a minimal example to reproduce the following Triton server inference issue: essentially, with the triton server hosting multiple model on one GPU, we see inconsistent result when GPU is heavily used

To reproduce this issue, I created a model repository with 2 TensorRT model:

- RensetV18
- InceptionV3

Both conversions are directly done using the NVIDIA torch2trt package (https://github.com/NVIDIA-AI-IOT/torch2trt)
```buildoutcfg
import torch
import torchvision

model = torchvision.models.inception_v3(pretrained=True).cuda().half().eval()
data = torch.randn((1, 3, 224, 224)).cuda().half()
with open('/workspace/ubuntu/model_repository_2011/resnet18/1/model', "wb") as f:
    f.write(model_trt.engine.serialize())
```

Launch Triton Server with this model repository, and run multiple jobs so that a significant portion of the GPU is running. One can see **for the same  model with same input, we have different results**



## Environment ##
I am using AWS g4dn.xlarge instance, it use T4 GPU.

To launch the triton server I use the latest NGC Triton Server container nvcr.io/nvidia/tritonserver:20.11-py3

To build the TensorRT models I use the latest NGC TensorRT container nvcr.io/nvidia/tensorrt:20.11-py3


## Setup ##

An example model repository is created. Download the tar file with following link 
```buildoutcfg
https://drive.google.com/file/d/1hBOpXMxSbeYbPltR23oJuzfdCxg6t1eo/view?usp=sharing
```

Uncompress the tar file, the directory `triton_issues_data` contains:

- `model_repository_2011`: the model repository we use to launch the triton server
- `testing_inputs.p` example input for both model


We run models using the python GRPC client therefore we download the following packages
```buildoutcfg
pip install tritonclient[http]
pip install nvidia-pyindex
pip install docopt 
```


# Reproducing the Issue ##

Suppose our head folder is `/home/ubuntu` 

We first launch the triton server 
```buildoutcfg
docker run -d --gpus=all --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v /home/ubuntu/trt_issues_data/model_repository_2011:/models -e CUDA_VISIBLE_DEVICES=0 nvcr.io/nvidia/tritonserver:20.11-py3 tritonserver --model-repository=/models --strict-model-config=false
```

We run both model `resnet18` and `inceptionv3` multiple times simultanously so that GPU is heavily used, on one T4 GPU machine:
- We run each model 4 times and save the result 
- Each time we use input size `64x3x224x224`, and repeat 64 times 

The folllowing commands do the aboves, and save resnet18 results in outputs1-4, and save the inceptionv3 results5-8 
```buildoutcfg
nohup python /home/ubuntu/tritonissues/triton_inference.py resnet18 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs1.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py inceptionv3 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs5.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py resnet18 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs2.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py inceptionv3 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs6.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py resnet18 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs3.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py inceptionv3 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs7.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py resnet18 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs4.p &
nohup python /home/ubuntu/tritonissues/triton_inference.py inceptionv3 /home/ubuntu/trt_issues_data/testing_inputs.p 64 outputs8.p &
```


One should excpect outputs 1-4 are the same, outputs 5-8 are the same. However, that is not the case, if you compare the results the discrepancy is significantly large.

```buildoutcfg
import numpy as np 
import pickle 

f1 = pickle.load(open('/home/ubuntu/outputs1.p','rb'))
f2 = pickle.load(open('/home/ubuntu/outputs2.p','rb'))
f3 = pickle.load(open('/home/ubuntu/outputs3.p','rb'))
f4 = pickle.load(open('/home/ubuntu/outputs4.p','rb'))


f5 = pickle.load(open('/home/ubuntu/outputs5.p','rb'))
f6 = pickle.load(open('/home/ubuntu/outputs6.p','rb'))
f7 = pickle.load(open('/home/ubuntu/outputs7.p','rb'))
f8 = pickle.load(open('/home/ubuntu/outputs8.p','rb'))

for entry in [f2,f3,f4]:
    print(np.max(np.abs(f1-entry)))

for entry in [f7,f6,f8]:
    print(np.max(np.abs(f5-entry)))
```

![plot](error.png)



You can reproduce the model_repository and inputs data yourself, like mentioned before, the two models are just TensorRT models from TorchVisions
- resnet18
- inceptionv3

The example inputs is just a random array
```buildoutcfg
np.random.rand(64,3,224,224)
```


















 