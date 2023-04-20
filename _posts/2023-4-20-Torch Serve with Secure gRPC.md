---
layout: post
title: TorchServe with GRPC Secure
---

Created: September 17, 2022 6:06 PM

In the previous [article](https://www.notion.so/TorchServe-with-GRPC-04c5cf2339d141b188cfe8ec95866187), I set out steps to serve a torch model using torchserve and communicate over gRPC. However, the communication between client and the server was over an insecure channel. This may not be an ideal scenario if you are working with any confidential data. This article will set out steps to communicate with torch serve with gRPC over a secure channel.

## Prerequisites

The prerequisite is that you have been through this [article](https://www.notion.so/TorchServe-with-GRPC-04c5cf2339d141b188cfe8ec95866187). Essentially, you should:

1. Have a .mar file generated
2. And compiled python output of the torchserve grpc proto files

To recap, the following is the file structure

```bash
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

## Generating the keys and certificates

- The Common Name (CN) is important. Apparently, the CN for server certificate must differ from CA certificate

[OpenSSL - error 18 at 0 depth lookup:self signed certificate](https://stackoverflow.com/a/19738223/8727339)

- In step 3, the CN value should be the domain name or IP address of the server. Here I have used 0.0.0.0 for generality.

```bash
rm *.pem
# create a CA's private key and certificate

export country="UK"
export state="London"
export location="London"
export ipaddress="0.0.0.0"
export email="test@gmail.com"
export org="Mo"
export org_unit="CV"

echo "1) Generate CA's private key and certificate"
# openssl req -x509 -newkey rsa:4096 -days 365 -keyout ca-key.pem -out ca-cert.pem

openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout ca-key.pem -out ca-cert.pem -subj "/C=${country}/ST=${state}/L=${location}/O=${org}/OU=${org_unit}/CN=random/emailAddress=${email}"

# The -x509 option is used to tell openssl to output a self-signed certificate instead of a certificate request. We need self signed because we are the CA. The private key is ca-key.pem and certificate is ca-cert.pem

# The above command will ask to provide certain identify information
echo "CA's self-signed certificate"
openssl x509 -in ca-cert.pem -noout -text

# /C is for Country
# /ST is for State or province
# /L is for Locality name or city
# /O is for Organisation
# /OU is for Organisation Unit
# /CN is for Common Name or domain name
# /emailAddress is for email address

echo "2) Generate web server's private key and CSR"
# Note: You can change the hostname parameter to the name or IP address of a server on your network, it just needs to match the server name that you connect to with the client.

# openssl req -newkey rsa:4096 -keyout server-key.pem -out server-req.pem

openssl req -newkey rsa:4096 -nodes -keyout server-key.pem -out server-req.pem -subj "/C=${country}/ST=${state}/L=${location}/O=${org}/OU=${org_unit}/CN=${ipaddress}/emailAddress=${email}"

# The above command will generated the encrypted private key and a CSR

echo "3) Sign server's CSR"

openssl x509 -req -in server-req.pem -days 365 -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -extfile server-ext.cnf

echo "4) Verifying certificate"

openssl verify -CAfile ca-cert.pem server-cert.pem
```

**`server-ext.cnf`**

```bash
subjectAltName=IP:0.0.0.0
```

## Editing Config.Properties

Change address to use https and set the enable_grpc_ssl, private_key_file and certificate_file

```bash
max_request_size=655350000
max_response_size=655350000
inference_address=https://0.0.0.0:8080
management_address=https://0.0.0.0:8081
metrics_address=https://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071

enable_grpc_ssl=true

private_key_file=torchserve_grpc/ssl/server-key.pem
certificate_file=torchserve_grpc/ssl/server-cert.pem
```

## Start the server

```bash
torchserve --start --model-store model_store --models all --ts-config configs/config.properties
```

## Check Models Served

```bash
curl https://0.0.0.0:8081/models --cacert torchserve_grpc/ssl/ca-cert.pem
```

In the `grpc_client.py` from previous [article](https://www.notion.so/TorchServe-with-GRPC-04c5cf2339d141b188cfe8ec95866187), add:

```python
def get_secure_management_stub():
    with open("torchserve_grpc/ssl/ca-cert.pem", 'rb') as f:
        creds= grpc.ssl_channel_credentials(f.read())
    channel = grpc.secure_channel('0.0.0.0:7071', credentials=creds)
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub

def get_secure_inference_stub():
    with open("torchserve_grpc/ssl/ca-cert.pem", 'rb') as f:
        creds= grpc.ssl_channel_credentials(f.read())
    channel = grpc.secure_channel('0.0.0.0:7070', credentials=creds)
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub

```

 

Test using

```python
if __name__ == '__main__':
    inference_stub = get_secure_inference_stub()
    management_stub = get_secure_management_stub()

    output = list_models(management_stub)
    print(f"output: {output.msg}")

    prediction = make_prediction(inference_stub, "torchserve_grpc/images/test.png")

    print(prediction)
```

Output

```bash
output: {
  "models": [
    {
      "modelName": "digitmodel",
      "modelUrl": "digit-model.mar"
    }
  ]
}
prediction: "8"
```

## Resources

[How to create & sign SSL/TLS certificates](https://dev.to/techschoolguru/how-to-create-sign-ssl-tls-certificates-2aai)

[https://github.com/joekottke/python-grpc-ssl](https://github.com/joekottke/python-grpc-ssl)

If you have any questions about this page, please feel free to [email](mailto:remotes.jackpot-0t@icloud.com) me. If this helped you, please consider [buying me a coffee](https://www.buymeacoffee.com/mohitmotwani)
