"""
Speed Up Quantized Model with TensorRT
======================================

Quantization algorithms quantize a deep learning model usually in a simulated way. That is, to simulate the effect of low-bit computation with float32 operators, the tensors are quantized to the targeted bit number and dequantized back to float32. Such a quantized model does not have any latency reduction. Thus, there should be a speedup stage to make the quantized model really accelerated with low-bit operators. 
This tutorial demonstrates how to accelerate a quantized model with `TensorRT <https://developer.nvidia.com/tensorrt>`_ as the inference engine in NNI. More inference engines will be supported in future release.

The process of speeding up a quantized model in NNI is that 1) the model with quantized weights and configuration is converted into onnx format, 2) the onnx model is fed into TensorRT to generate an inference engine. The engine is used for low latency model inference.

There are two modes of the speedup: 1) leveraging post-training quantization of TensorRT, 2) using TensorRT as a pure acceleration backend. The two modes will be explained in the usage section below.

Prerequisite
------------
When using TensorRT to speed up a quantized model, you are highly recommended to use the PyTorch docker image provided by NVIDIA.
Users can refer to `this web page <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__ for detailed usage of the docker image.
The docker image "nvcr.io/nvidia/pytorch:22.09-py3" has been tested for the quantization speedup in NNI.

An example command to launch the docker container is `nvidia-docker run -it nvcr.io/nvidia/pytorch:22.09-py3`.
In the docker image, users should install nni>=3.0, pytorch_lightning, pycuda.

Usage
-----

Mode #1: Leveraging post-training quantization of TensorRT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As TensorRT has supported post-training quantization, directly leveraging this functionality is a natural way to use TensorRT. This mode is called "with calibration data". In this mode, the quantization-aware training algorithms (e.g., `QAT <https://nni.readthedocs.io/en/stable/reference/compression/quantizer.html#qat-quantizer>`_, `LSQ <https://nni.readthedocs.io/en/stable/reference/compression/quantizer.html#lsq-quantizer>`_) only take charge of adjusting model weights to be more quantization friendly, and leave the last-step quantization to the post-training quantization of TensorRT.

"""
import torch
import torchvision
import torchvision.transforms as transforms
skip_exec = True
if not skip_exec:

    def prepare_data_loaders(data_path, batch_size):
        if False:
            print('Hello World!')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dataset = torchvision.datasets.ImageNet(data_path, split='train', transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
        sampler = torch.utils.data.SequentialSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        return data_loader
    data_path = '/data'
    data_loader = prepare_data_loaders(data_path, batch_size=128)
    calib_data = None
    for (image, target) in data_loader:
        calib_data = image.numpy()
        break
    from nni.compression.quantization_speedup.calibrator import Calibrator
    calib = Calibrator(calib_data, 'data/calib_cache_file.cache', batch_size=64)
if not skip_exec:
    from nni_assets.compression.mobilenetv2 import MobileNetV2
    model = MobileNetV2()
    float_model_file = 'mobilenet_pretrained_float.pth'
    state_dict = torch.load(float_model_file)
    model.load_state_dict(state_dict)
    model.eval()
if not skip_exec:
    from nni.compression.quantization_speedup import ModelSpeedupTensorRT
    engine = ModelSpeedupTensorRT(model, input_shape=(64, 3, 224, 224))
    engine.compress_with_calibrator(calib)
if not skip_exec:
    from nni_assets.compression.mobilenetv2 import AverageMeter, accuracy
    import time

    def test_accelerated_model(engine, data_loader, neval_batches):
        if False:
            print('Hello World!')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        cnt = 0
        total_time = 0
        for (image, target) in data_loader:
            start_time = time.time()
            (output, time_span) = engine.inference(image)
            infer_time = time.time() - start_time
            print('time: ', time_span, infer_time)
            total_time += time_span
            start_time = time.time()
            output = output.view(-1, 1000)
            cnt += 1
            (acc1, acc5) = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            rest_time = time.time() - start_time
            print('rest time: ', rest_time)
            if cnt >= neval_batches:
                break
        print('inference time: ', total_time / neval_batches)
        return (top1, top5)
    data_loader = prepare_data_loaders(data_path, batch_size=64)
    (top1, top5) = test_accelerated_model(engine, data_loader, neval_batches=32)
    print('Accuracy of mode #1: ', top1, top5)
'\n\nMode #2: Using TensorRT as a pure acceleration backend\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\nIn this mode, the post-training quantization within TensorRT is not used, instead, the quantization bit-width and the range of tensor values are fed into TensorRT for speedup (i.e., with `trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS` configured).\n\n'
if not skip_exec:
    model = MobileNetV2()
    state_dict = torch.load(float_model_file)
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device('cuda')
    model.to(device)
if not skip_exec:
    from nni_assets.compression.mobilenetv2 import evaluate
    from nni.compression.utils import TorchEvaluator
    data_loader = prepare_data_loaders(data_path, batch_size=128)

    def eval_for_calibration(model):
        if False:
            for i in range(10):
                print('nop')
        evaluate(model, data_loader, neval_batches=1, device=device)
    dummy_input = torch.Tensor(64, 3, 224, 224).to(device)
    predict_func = TorchEvaluator(predicting_func=eval_for_calibration, dummy_input=dummy_input)
from nni.compression.quantization import PtqQuantizer
if not skip_exec:
    config_list = [{'quant_types': ['input', 'weight', 'output'], 'quant_bits': {'input': 8, 'weight': 8, 'output': 8}, 'quant_dtype': 'int', 'quant_scheme': 'per_tensor_symmetric', 'op_types': ['default']}]
    quantizer = PtqQuantizer(model, config_list, predict_func, True)
    quantizer.compress()
    calibration_config = quantizer.export_model()
    print('quant result config: ', calibration_config)
if not skip_exec:
    model = MobileNetV2()
    state_dict = torch.load(float_model_file)
    model.load_state_dict(state_dict)
    model.eval()
    engine = ModelSpeedupTensorRT(model, input_shape=(64, 3, 224, 224), config=calibration_config)
    engine.compress()
    data_loader = prepare_data_loaders(data_path, batch_size=64)
    (top1, top5) = test_accelerated_model(engine, data_loader, neval_batches=32)
    print('Accuracy of mode #2: ', top1, top5)