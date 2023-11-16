"""
Title: Serving TensorFlow models with TFServing
Author: [Dimitre Oliveira](https://www.linkedin.com/in/dimitre-oliveira-7a1a0113a/)
Date created: 2023/01/02
Last modified: 2023/01/02
Description: How to serve TensorFlow models with TensorFlow Serving.
Accelerator: None
"""
'\n## Introduction\n\nOnce you build a machine learning model, the next step is to serve it.\nYou may want to do that by exposing your model as an endpoint service.\nThere are many frameworks that you can use to do that, but the TensorFlow\necosystem has its own solution called\n[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).\n\nFrom the TensorFlow Serving\n[GitHub page](https://github.com/tensorflow/serving):\n\n> TensorFlow Serving is a flexible, high-performance serving system for machine\nlearning models, designed for production environments. It deals with the\ninference aspect of machine learning, taking models after training and\nmanaging their lifetimes, providing clients with versioned access via a\nhigh-performance, reference-counted lookup table. TensorFlow Serving provides\nout-of-the-box integration with TensorFlow models, but can be easily extended\nto serve other types of models and data."\n\nTo note a few features:\n\n- It can serve multiple models, or multiple versions of the same model\nsimultaneously\n- It exposes both gRPC as well as HTTP inference endpoints\n- It allows deployment of new model versions without changing any client code\n- It supports canarying new versions and A/B testing experimental models\n- It adds minimal latency to inference time due to efficient, low-overhead\nimplementation\n- It features a scheduler that groups individual inference requests into batches\nfor joint execution on GPU, with configurable latency controls\n- It supports many servables: Tensorflow models, embeddings, vocabularies,\nfeature transformations and even non-Tensorflow-based machine learning models\n\nThis guide creates a simple [MobileNet](https://arxiv.org/abs/1704.04861)\nmodel using the [Keras applications API](https://keras.io/api/applications/),\nand then serves it with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).\nThe focus is on TensorFlow Serving, rather than the modeling and training in\nTensorFlow.\n\n> Note: you can find a Colab notebook with the full working code at\n[this link](https://colab.research.google.com/drive/1nwuIJa4so1XzYU0ngq8tX_-SGTO295Mu?usp=sharing).\n'
'\n## Dependencies\n'
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import json
import shutil
import requests
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
'\n## Model\n\nHere we load a pre-trained [MobileNet](https://arxiv.org/abs/1704.04861)\nfrom the [Keras applications](https://keras.io/api/applications/), this is the\nmodel that we are going to serve.\n'
model = keras.applications.MobileNet()
"\n## Preprocessing\n\nMost models don't work out of the box on raw data, they usually require some\nkind of preprocessing step to adjust the data to the model requirements,\nin the case of this MobileNet we can see from its\n[API page](https://keras.io/api/applications/mobilenet/) that it requires\nthree basic steps for its input images:\n\n- Pixel values normalized to the `[0, 1]` range\n- Pixel values scaled to the `[-1, 1]` range\n- Images with the shape of `(224, 224, 3)` meaning `(height, width, channels)`\n\nWe can do all of that with the following function:\n"

def preprocess(image, mean=0.5, std=0.5, shape=(224, 224)):
    if False:
        print('Hello World!')
    'Scale, normalize and resizes images.'
    image = image / 255.0
    image = (image - mean) / std
    image = tf.image.resize(image, shape)
    return image
'\n**A note regarding preprocessing and postprocessing using the "keras.applications" API**\n\nAll models that are available at the [Keras applications](https://keras.io/api/applications/)\nAPI also provide `preprocess_input` and `decode_predictions` functions, those\nfunctions are respectively responsible for the preprocessing and postprocessing\nof each model, and already contains all the logic necessary for those steps.\nThat is the recommended way to process inputs and outputs when using Keras\napplications models.\nFor this guide, we are not using them to present the advantages of custom\nsignatures in a clearer way.\n'
'\n## Postprocessing\n\nIn the same context most models output values that need extra processing to\nmeet the user requirements, for instance, the user does not want to know the\nlogits values for each class given an image, what the user wants is to know\nfrom which class it belongs. For our model, this translates to the following\ntransformations on top of the model outputs:\n\n- Get the index of the class with the highest prediction\n- Get the name of the class from that index\n'
imagenet_labels_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
response = requests.get(imagenet_labels_url)
labels = [x for x in response.text.split('\n') if x != ''][1:]
tf_labels = tf.constant(labels, dtype=tf.string)

def postprocess(prediction, labels=tf_labels):
    if False:
        while True:
            i = 10
    'Convert from probs to labels.'
    indices = tf.argmax(prediction, axis=-1)
    label = tf.gather(params=labels, indices=indices)
    return label
"\nNow let's download a banana picture and see how everything comes together.\n"
response = requests.get('https://i.imgur.com/j9xCCzn.jpeg', stream=True)
with open('banana.jpeg', 'wb') as f:
    shutil.copyfileobj(response.raw, f)
sample_img = plt.imread('./banana.jpeg')
print(f'Original image shape: {sample_img.shape}')
print(f'Original image pixel range: ({sample_img.min()}, {sample_img.max()})')
plt.imshow(sample_img)
plt.show()
preprocess_img = preprocess(sample_img)
print(f'Preprocessed image shape: {preprocess_img.shape}')
print(f'Preprocessed image pixel range: ({preprocess_img.numpy().min()},', f'{preprocess_img.numpy().max()})')
batched_img = tf.expand_dims(preprocess_img, axis=0)
batched_img = tf.cast(batched_img, tf.float32)
print(f'Batched image shape: {batched_img.shape}')
model_outputs = model(batched_img)
print(f'Model output shape: {model_outputs.shape}')
print(f'Predicted class: {postprocess(model_outputs)}')
'\n## Save the model\n\nTo load our trained model into TensorFlow Serving, we first need to save it in\n[SavedModel](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model)\nformat. This will create a protobuf file in a well-defined directory hierarchy,\nand will include a version number.\n[TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) allows us\nto select which version of a model, or "servable" we want to use when we make\ninference requests. Each version will be exported to a different sub-directory\nunder the given path.\n'
model_dir = './model'
model_version = 1
model_export_path = f'{model_dir}/{model_version}'
tf.saved_model.save(model, export_dir=model_export_path)
print(f'SavedModel files: {os.listdir(model_export_path)}')
"\n## Examine your saved model\n\nWe'll use the command line utility `saved_model_cli` to look at the\n[MetaGraphDefs](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/MetaGraphDef)\n(the models) and [SignatureDefs](https://www.tensorflow.org/tfx/serving/signature_defs)\n(the methods you can call) in our SavedModel. See\n[this discussion of the SavedModel CLI](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel)\nin the TensorFlow Guide.\n"
'shell\nsaved_model_cli show --dir {model_export_path} --tag_set serve --signature_def serving_default\n'
"\nThat tells us a lot about our model! For instance, we can see that its inputs\nhave a 4D shape `(-1, 224, 224, 3)` which means\n`(batch_size, height, width, channels)`, also note that this model requires a\nspecific image shape `(224, 224, 3)` this means that we may need to reshape\nour images before sending them to the model. We can also see that the model's\noutputs have a `(-1, 1000)` shape which are the logits for the 1000 classes of\nthe [ImageNet](https://www.image-net.org) dataset.\n\nThis information doesn't tell us everything, like the fact that the pixel\nvalues needs to be in the `[-1, 1]` range, but it's a great start.\n\n## Serve your model with TensorFlow Serving\n\n### Install TFServing\n\nWe're preparing to install TensorFlow Serving using\n[Aptitude](https://wiki.debian.org/Aptitude) since this Colab runs in a Debian\nenvironment. We'll add the `tensorflow-model-server` package to the list of\npackages that Aptitude knows about. Note that we're running as root.\n\n\n> Note: This example is running TensorFlow Serving natively, but [you can also\nrun it in a Docker container](https://www.tensorflow.org/tfx/serving/docker),\nwhich is one of the easiest ways to get started using TensorFlow Serving.\n\n```shell\nwget 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-universal-2.8.0/t/tensorflow-model-server-universal/tensorflow-model-server-universal_2.8.0_all.deb'\ndpkg -i tensorflow-model-server-universal_2.8.0_all.deb\n```\n"
"\n### Start running TensorFlow Serving\n\nThis is where we start running TensorFlow Serving and load our model. After it\nloads, we can start making inference requests using REST. There are some\nimportant parameters:\n\n- `port`: The port that you'll use for gRPC requests.\n- `rest_api_port`: The port that you'll use for REST requests.\n- `model_name`: You'll use this in the URL of REST requests. It can be\nanything.\n- `model_base_path`: This is the path to the directory where you've saved your\nmodel.\n\nCheck the [TFServing API reference](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/model_servers/main.cc)\nto get all the parameters available.\n"
os.environ['MODEL_DIR'] = f'{model_dir}'
"\n```shell\n%%bash --bg\nnohup tensorflow_model_server   --port=8500   --rest_api_port=8501   --model_name=model   --model_base_path=$MODEL_DIR >server.log 2>&1\n```\n\n```shell\n# We can check the logs to the server to help troubleshooting\n!cat server.log\n```\noutputs:\n```\n[warn] getaddrinfo: address family for nodename not supported\n[evhttp_server.cc : 245] NET_LOG: Entering the event loop ...\n```\n\n```shell\n# Now we can check if tensorflow is in the active services\n!sudo lsof -i -P -n | grep LISTEN\n```\noutputs:\n```\nnode         7 root   21u  IPv6  19100      0t0  TCP *:8080 (LISTEN)\nkernel_ma   34 root    7u  IPv4  18874      0t0  TCP 172.28.0.12:6000 (LISTEN)\ncolab-fil   63 root    5u  IPv4  17975      0t0  TCP *:3453 (LISTEN)\ncolab-fil   63 root    6u  IPv6  17976      0t0  TCP *:3453 (LISTEN)\njupyter-n   81 root    6u  IPv4  18092      0t0  TCP 172.28.0.12:9000 (LISTEN)\npython3    101 root   23u  IPv4  18252      0t0  TCP 127.0.0.1:44915 (LISTEN)\npython3    132 root    3u  IPv4  20548      0t0  TCP 127.0.0.1:15264 (LISTEN)\npython3    132 root    4u  IPv4  20549      0t0  TCP 127.0.0.1:37977 (LISTEN)\npython3    132 root    9u  IPv4  20662      0t0  TCP 127.0.0.1:40689 (LISTEN)\ntensorflo 1101 root    5u  IPv4  35543      0t0  TCP *:8500 (LISTEN)\ntensorflo 1101 root   12u  IPv4  35548      0t0  TCP *:8501 (LISTEN)\n```\n\n## Make a request to your model in TensorFlow Serving\n\nNow let's create the JSON object for an inference request, and see how well\nour model classifies it:\n\n### REST API\n\n#### Newest version of the servable\n\nWe'll send a predict request as a POST to our server's REST endpoint, and pass\nit as an example. We'll ask our server to give us the latest version of our\nservable by not specifying a particular version.\n"
data = json.dumps({'signature_name': 'serving_default', 'instances': batched_img.numpy().tolist()})
url = 'http://localhost:8501/v1/models/model:predict'

def predict_rest(json_data, url):
    if False:
        while True:
            i = 10
    json_response = requests.post(url, data=json_data)
    response = json.loads(json_response.text)
    rest_outputs = np.array(response['predictions'])
    return rest_outputs
'\n```python\nrest_outputs = predict_rest(data, url)\n\nprint(f"REST output shape: {rest_outputs.shape}")\nprint(f"Predicted class: {postprocess(rest_outputs)}")\n```\n\noutputs:\n```\nREST output shape: (1, 1000)\nPredicted class: [b\'banana\']\n```\n\n### gRPC API\n\n[gRPC](https://grpc.io/) is based on the Remote Procedure Call (RPC) model and\nis a technology for implementing RPC APIs that uses HTTP 2.0 as its underlying\ntransport protocol. gRPC is usually preferred for low-latency, highly scalable,\nand distributed systems. If you wanna know more about the REST vs gRPC\ntradeoffs, checkout\n[this article](https://cloud.google.com/blog/products/api-management/understanding-grpc-openapi-and-rest-and-when-to-use-them).\n'
import grpc
channel = grpc.insecure_channel('localhost:8500')
'\n```shell\npip install -q tensorflow_serving_api\n```\n\n```python\nfrom tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc\n\n# Create a stub made for prediction\n# This stub will be used to send the gRPCrequest to the TF Server\nstub = prediction_service_pb2_grpc.PredictionServiceStub(channel)\n```\n'
loaded_model = tf.saved_model.load(model_export_path)
input_name = list(loaded_model.signatures['serving_default'].structured_input_signature[1].keys())[0]
'\n```python\ndef predict_grpc(data, input_name, stub):\n    # Create a gRPC request made for prediction\n    request = predict_pb2.PredictRequest()\n\n    # Set the name of the model, for this use case it is "model"\n    request.model_spec.name = "model"\n\n    # Set which signature is used to format the gRPC query\n    # here the default one "serving_default"\n    request.model_spec.signature_name = "serving_default"\n\n    # Set the input as the data\n    # tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor\n    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(data.numpy().tolist()))\n\n    # Send the gRPC request to the TF Server\n    result = stub.Predict(request)\n    return result\n\n\ngrpc_outputs = predict_grpc(batched_img, input_name, stub)\ngrpc_outputs = np.array([grpc_outputs.outputs[\'predictions\'].float_val])\n\nprint(f"gRPC output shape: {grpc_outputs.shape}")\nprint(f"Predicted class: {postprocess(grpc_outputs)}")\n```\n\noutputs:\n```\ngRPC output shape: (1, 1000)\nPredicted class: [b\'banana\']\n```\n'
'\n## Custom signature\n\nNote that for this model we always need to preprocess and postprocess all\nsamples to get the desired output, this can get quite tricky if are\nmaintaining and serving several models developed by a large team, and each one\nof them might require different processing logic.\n\nTensorFlow allows us to customize the model graph to embed all of that\nprocessing logic, which makes model serving much easier, there are different\nways to achieve this, but since we are going to server the models using\nTFServing we can customize the model graph straight into the serving signature.\n\nWe can just use the following code to export the same model that already\ncontains the preprocessing and postprocessing logic as the default signature,\nthis allows this model to make predictions on raw data.\n'

def export_model(model, labels):
    if False:
        print('Hello World!')

    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
    def serving_fn(image):
        if False:
            while True:
                i = 10
        processed_img = preprocess(image)
        probs = model(processed_img)
        label = postprocess(probs)
        return {'label': label}
    return serving_fn
model_sig_version = 2
model_sig_export_path = f'{model_dir}/{model_sig_version}'
tf.saved_model.save(model, export_dir=model_sig_export_path, signatures={'serving_default': export_model(model, labels)})
'shell\nsaved_model_cli show --dir {model_sig_export_path} --tag_set serve --signature_def serving_default\n'
"\nNote that this model has a different signature, its input is still 4D but now\nwith a `(-1, -1, -1, 3)` shape, which means that it supports images with any\nheight and width size. Its output also has a different shape, it no longer\noutputs the 1000-long logits.\n\nWe can test the model's prediction using a specific signature using this API\nbelow:\n"
batched_raw_img = tf.expand_dims(sample_img, axis=0)
batched_raw_img = tf.cast(batched_raw_img, tf.float32)
loaded_model = tf.saved_model.load(model_sig_export_path)
loaded_model.signatures['serving_default'](**{'image': batched_raw_img})
"\n## Prediction using a particular version of the servable\n\nNow let's specify a particular version of our servable. Note that when we\nsaved the model with a custom signature we used a different folder, the first\nmodel was saved in folder `/1` (version 1), and the one with a custom\nsignature in folder `/2` (version 2). By default, TFServing will serve all\nmodels that share the same base parent folder.\n\n### REST API\n"
data = json.dumps({'signature_name': 'serving_default', 'instances': batched_raw_img.numpy().tolist()})
url_sig = 'http://localhost:8501/v1/models/model/versions/2:predict'
'\n```python\nprint(f"REST output shape: {rest_outputs.shape}")\nprint(f"Predicted class: {rest_outputs}")\n```\n\noutputs:\n```\nREST output shape: (1,)\nPredicted class: [\'banana\']\n```\n\n### gRPC API\n'
'\n```python\nchannel = grpc.insecure_channel("localhost:8500")\nstub = prediction_service_pb2_grpc.PredictionServiceStub(channel)\n```\n'
input_name = list(loaded_model.signatures['serving_default'].structured_input_signature[1].keys())[0]
'\n```python\ngrpc_outputs = predict_grpc(batched_raw_img, input_name, stub)\ngrpc_outputs = np.array([grpc_outputs.outputs[\'label\'].string_val])\n\nprint(f"gRPC output shape: {grpc_outputs.shape}")\nprint(f"Predicted class: {grpc_outputs}")\n```\n\noutputs:\n\n```\ngRPC output shape: (1, 1)\nPredicted class: [[b\'banana\']]\n```\n\n## Additional resources\n\n- [Colab notebook with the full working code](https://colab.research.google.com/drive/1nwuIJa4so1XzYU0ngq8tX_-SGTO295Mu?usp=sharing)\n- [Train and serve a TensorFlow model with TensorFlow Serving - TensorFlow blog](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#make_a_request_to_your_model_in_tensorflow_serving)\n- [TensorFlow Serving playlist - TensorFlow YouTube channel](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwHdpVQVohY7-qcYf2s1UYK)\n'