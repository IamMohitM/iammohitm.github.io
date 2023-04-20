---
layout: post
title: TorchServe with GRPC
---

Created: June 4, 2022 11:27 AM

This page documents a template for serving torch models and communicating with them with GRPC. I believe the content available online mostly focuses on using HTTP for communicating with the served models. Therefore, this is a good place for someone who has a preference for faster inference with the served models using GRPC. 

This page does not explain the theory behind TorchServe or GRPC. Instead, it sets the steps for serving the models and implementing a grpc client.  

## Directory structure

```python
project_root_folder
|__torchserve_grpc
	|__ model_store
		|__digit_model.mar
	|__model_weights
		|__digitcnn_state_dict.pth
	|__images
		|__test.png
	|__handler.py
	|__model.py
	|__inference.proto
	|__management.proto
	|__inference_pb2.py
	|__inference_pb2_grpc.py
	|__management_pb2.py
	|__management_pb2_grpc.py
  |__grpc_client.py

```

## Model Definition file (model.py)

```jsx
import torch, torchvision

class DigitCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self._model_prep()

    def _model_prep(self):  
        model = torchvision.models.mobilenet_v3_small(pretrained=True, progress=True, )
        classification_layer = torch.nn.Sequential(
            torch.nn.Linear(576, out_features=1024),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10),
            torch.nn.Softmax()
        )
        model.classifier = classification_layer

        for param in model.features[:11].parameters():
            param.requires_grad = False
        return model

    def forward(self, x):
        return self.model(x)
```

## Custom Handler (handler.py)

```python
from ts.torch_handler.vision_handler import VisionHandler
import torch
from PIL import Image
from torchvision import transforms
import logging
import io
import base64

class CustomHandler(VisionHandler):
    def __init__(self):
        super(CustomHandler, self).__init__()
        self.image_processing = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()])
    
    def preprocess(self, data):
        images = []
        for row in data:
            # Compat layer: normally the envelope should just return the data
            # directly, but older versions of Torchserve didn't have envelope.
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)

            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
                image = self.image_processing(image)
                logging.info(f"image shape after preprocess: {image.shape}")
            else:
                # if the image is a list
                image = torch.FloatTensor(image)
            
            images.append(image)

        return torch.stack(images).to(self.device)
    
    def postprocess(self, data):
        logging.info("Inside Post Process")
        logging.info(f"Outputs: {data}")
        predictions = torch.argmax(data, axis=1) + 1
        logging.info(predictions.tolist())
        return predictions.tolist()
```

## Archiving the model

```python
torch-model-archiver --model-name digit-model --export-path torchserve_grpc/model_store --version 1.0 --model-file torchserve_grpc/model.py --serialized-file torchserve_grpc/model_weights/digitcnn_state_dict.pth --handler torchserve_grpc/handler.py --force
```

## Serving the model

```python
torchserve --start --model-store torchserve_grpc/model_store --models digitmodel=digit-model.mar --no-config-snapshot
```

## Protocol buffer files

[serve/inference.proto at master · pytorch/serve](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto)

`inference.proto`

```protobuf
syntax = "proto3";
package org.pytorch.serve.grpc.inference;

import "google/protobuf/empty.proto";

option java_multiple_files = true;

message PredictionsRequest {
    // Name of model.
    string model_name = 1; //required

    // Version of model to run prediction on.
    string model_version = 2; //optional

    // input data for model prediction
    map<string, bytes> input = 3; //required
}

message PredictionResponse {
    // TorchServe health
    bytes prediction = 1;
}

message TorchServeHealthResponse {
    // TorchServe health
    string health = 1;
}

service InferenceAPIsService {
    rpc Ping(google.protobuf.Empty) returns (TorchServeHealthResponse) {}

    // Predictions entry point to get inference using default model version.
    rpc Predictions(PredictionsRequest) returns (PredictionResponse) {}
}
```

[serve/management.proto at master · pytorch/serve](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto)

`management.proto`

```protobuf
syntax = "proto3";

package org.pytorch.serve.grpc.management;

option java_multiple_files = true;

message ManagementResponse {
    // Response string of different management API calls.
    string msg = 1;
}

message DescribeModelRequest {
    // Name of model to describe.
    string model_name = 1; //required
    // Version of model to describe.
    string model_version = 2; //optional
    // Customized metadata
    bool customized = 3; //optional
}

message ListModelsRequest {
    // Use this parameter to specify the maximum number of items to return. When this value is present, TorchServe does not return more than the specified number of items, but it might return fewer. This value is optional. If you include a value, it must be between 1 and 1000, inclusive. If you do not include a value, it defaults to 100.
    int32 limit = 1; //optional

    // The token to retrieve the next set of results. TorchServe provides the token when the response from a previous call has more results than the maximum page size.
    int32 next_page_token = 2; //optional
}

message RegisterModelRequest {
    // Inference batch size, default: 1.
    int32 batch_size = 1; //optional

    // Inference handler entry-point. This value will override handler in MANIFEST.json if present.
    string handler = 2; //optional

    // Number of initial workers, default: 0.
    int32 initial_workers = 3; //optional

    // Maximum delay for batch aggregation, default: 100.
    int32 max_batch_delay = 4; //optional

    // Name of model. This value will override modelName in MANIFEST.json if present.
    string model_name = 5; //optional

    // Maximum time, in seconds, the TorchServe waits for a response from the model inference code, default: 120.
    int32 response_timeout = 6; //optional

    // Runtime for the model custom service code. This value will override runtime in MANIFEST.json if present.
    string runtime = 7; //optional

    // Decides whether creation of worker synchronous or not, default: false.
    bool synchronous = 8; //optional

    // Model archive download url, support local file or HTTP(s) protocol.
    string url = 9; //required

    // Decides whether S3 SSE KMS enabled or not, default: false.
    bool s3_sse_kms = 10; //optional
}

message ScaleWorkerRequest {

    // Name of model to scale workers.
    string model_name = 1; //required

    // Model version.
    string model_version = 2; //optional

    // Maximum number of worker processes.
    int32 max_worker = 3; //optional

    // Minimum number of worker processes.
    int32 min_worker = 4; //optional

    // Number of GPU worker processes to create.
    int32 number_gpu = 5; //optional

    // Decides whether the call is synchronous or not, default: false.
    bool synchronous = 6; //optional

    // Waiting up to the specified wait time if necessary for a worker to complete all pending requests. Use 0 to terminate backend worker process immediately. Use -1 for wait infinitely.
    int32 timeout = 7; //optional
}

message SetDefaultRequest {
    // Name of model whose default version needs to be updated.
    string model_name = 1; //required

    // Version of model to be set as default version for the model
    string model_version = 2; //required
}

message UnregisterModelRequest {
    // Name of model to unregister.
    string model_name = 1; //required

    // Name of model to unregister.
    string model_version = 2; //optional
}

service ManagementAPIsService {
    // Provides detailed information about the default version of a model.
    rpc DescribeModel(DescribeModelRequest) returns (ManagementResponse) {}

    // List registered models in TorchServe.
    rpc ListModels(ListModelsRequest) returns (ManagementResponse) {}

    // Register a new model in TorchServe.
    rpc RegisterModel(RegisterModelRequest) returns (ManagementResponse) {}

    // Configure number of workers for a default version of a model.This is a asynchronous call by default. Caller need to call describeModel to check if the model workers has been changed.
    rpc ScaleWorker(ScaleWorkerRequest) returns (ManagementResponse) {}

    // Set default version of a model
    rpc SetDefault(SetDefaultRequest) returns (ManagementResponse) {}

    // Unregister the default version of a model from TorchServe if it is the only version available.This is a asynchronous call by default. Caller can call listModels to confirm model is unregistered
    rpc UnregisterModel(UnregisterModelRequest) returns (ManagementResponse) {}
}
```

python files can be generated via protocol buffer compiler

```protobuf
python -m grpc_tools.protoc --proto_path=/Users/mo/Projects/Blog/torchserve_grpc/ --python_out=torchserve_grpc --grpc_python_out=torchserve_grpc torchserve_grpc/management.proto torchserve_grpc/inference.proto 
```

The above command should generate 4 python files: 

- `inference_pb2`
- `inference_pb2_grpc`
- `management_pb2`
- `management_pb2_grpc`

The above files are needed to construct the objects and call the services implemented by the grpc server. The grpc server is already implemented and running with torchserve. 

## Writing a gRPC client

1. We need a stub to make the requests to the server
2. The gRPC server is already implement by torchserve and starts along with torchserve
3. Based on the `inference.proto` and `management.proto` files we can figure out which functions can be called along with their required parameters

We’ll try to construct the gRPC client that is shared by the [official documentation](https://pytorch.org/serve/grpc_api.html). The client file can be found [here](https://github.com/pytorch/serve/blob/master/ts_scripts/torchserve_grpc_client.py).

First, we import the generated files in the `grpc_client.py`

```python
import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc
import json
```

By default ports 7071 and 7071 are exposed for inference and management, respectively. Therefore we may need to two stubs - one for inference and another for management requests.

```python
def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub

def get_management_stub():
    channel = grpc.insecure_channel('localhost:7071')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub
```

Note that you can replace [localhost](http://localhost) with the private ip (or another remotely accessible ip) of a remote server. 

You may also notice above that while making the above stub, we use the gRPC modules generated by the protocol buffer compiler. For example, the management stub can now make the rpc calls that have been defined by the management.proto files.

Now, we can write any function that makes a request to a service exposed by the torchserve. The exposed services are given by the protocol buffer files (.proto) above. Let’s write a function that will list all the models that are served.

We can notice that the `management.proto` file have a `ListModels` RPC which takes in a parameter `ListModelsRequest` message also defined in the same file. The message object can be created via the management_pb2 file and the rpc can be made with the stub. 

```python
def list_models(management_stub):
    list_model_request_object = management_pb2.ListModelsRequest(limit=10)
    return management_stub.ListModels(list_model_request_object)
```

Testing this:

```python
if __name__ == '__main__':

    inference_stub = get_inference_stub()
    management_stub = get_management_stub()

    output = list_models(management_stub)
    print(f"output: {output}")

    message = json.loads(output.msg)
    print(f"message: {message}")
```

Output:

```python
output: msg: "{\n  \"models\": [\n    {\n      \"modelName\": \"digitmodel\",\n      \"modelUrl\": \"digit-model.mar\"\n    }\n  ]\n}"

message: {'models': [{'modelName': 'digitmodel', 'modelUrl': 'digit-model.mar'}]}
```

Finally, if we want to make a prediction, the request message and rpc are defined in the inference.proto file. We’d like to make PredictionsRequest object with inference_pb2 which needs the model name, model_version and bytes input. 

```python

def make_prediction(inference_stub, image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    input_data = {"data": image_bytes}
    prediction_request = inference_pb2.PredictionsRequest(model_name="digitmodel", input=input_data)
    
    prediction = inference_stub.Predictions(prediction_request)
    return prediction
```

Testing this:

```python
if __name__ == '__main__':
    inference_stub = get_inference_stub()
    management_stub = get_management_stub()

    prediction = make_prediction(inference_stub, "torchserve_grpc/images/test.png")

    print(prediction)
```

Output:

```python
prediction: "8"
```

If you have any questions about this page, please feel free to [email](mailto:remotes.jackpot-0t@icloud.com) me. If this helped you, please consider [buying me a coffee](https://www.buymeacoffee.com/mohitmotwani)