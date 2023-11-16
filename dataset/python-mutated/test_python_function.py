import cv2
import glob
import numpy
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import random
import tempfile
import time
from PIL import Image, ImageEnhance
from nvidia.dali.ops import _DataNode
from nvidia.dali.pipeline import Pipeline
from nose_utils import raises
from test_utils import get_dali_extra_path
test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')

def test_dlpack_conversions():
    if False:
        i = 10
        return i + 15
    array = numpy.arange(0, 10, 0.5)
    reshaped = array.reshape((2, 10, 1))
    slice = reshaped[:, 2:5, :]
    dlpack = ops._dlpack_from_array(slice)
    result_array = ops._dlpack_to_array(dlpack)
    assert result_array.shape == slice.shape
    assert numpy.array_equal(result_array, slice)

def resize(image):
    if False:
        print('Hello World!')
    return numpy.array(Image.fromarray(image).resize((300, 300)))

class CommonPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, device_id, _seed, image_dir, prefetch_queue_depth=2):
        if False:
            while True:
                i = 10
        super().__init__(batch_size, num_threads, device_id, seed=_seed, prefetch_queue_depth=prefetch_queue_depth)
        self.input = ops.readers.File(file_root=image_dir)
        self.decode = ops.decoders.Image(device='cpu', output_type=types.RGB)
        self.resize = ops.PythonFunction(function=resize, output_layouts='HWC')

    def load(self):
        if False:
            while True:
                i = 10
        (jpegs, labels) = self.input()
        decoded = self.decode(jpegs)
        resized = self.resize(decoded)
        return (resized, labels)

    def define_graph(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class BasicPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        if False:
            return 10
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)

    def define_graph(self):
        if False:
            while True:
                i = 10
        (images, labels) = self.load()
        return images

class PythonOperatorPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function, prefetch_queue_depth=2):
        if False:
            return 10
        super().__init__(batch_size, num_threads, device_id, seed, image_dir, prefetch_queue_depth=prefetch_queue_depth)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        if False:
            while True:
                i = 10
        (images, labels) = self.load()
        processed = self.python_function(images)
        assert isinstance(processed, _DataNode)
        return processed

class FlippingPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.flip = ops.Flip(horizontal=1)

    def define_graph(self):
        if False:
            while True:
                i = 10
        (images, labels) = self.load()
        flipped = self.flip(images)
        return flipped

class TwoOutputsPythonOperatorPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function, op=ops.PythonFunction):
        if False:
            while True:
                i = 10
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.python_function = op(function=function, num_outputs=2)

    def define_graph(self):
        if False:
            while True:
                i = 10
        (images, labels) = self.load()
        (out1, out2) = self.python_function(images)
        assert isinstance(out1, _DataNode)
        assert isinstance(out2, _DataNode)
        return (out1, out2)

class MultiInputMultiOutputPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function, batch_processing=False):
        if False:
            i = 10
            return i + 15
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(function=function, num_outputs=3, batch_processing=batch_processing)

    def define_graph(self):
        if False:
            return 10
        (images1, labels1) = self.load()
        (images2, labels2) = self.load()
        (out1, out2, out3) = self.python_function(images1, images2)
        assert isinstance(out1, _DataNode)
        assert isinstance(out2, _DataNode)
        assert isinstance(out3, _DataNode)
        return (out1, out2, out3)

class DoubleLoadPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        if False:
            i = 10
            return i + 15
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)

    def define_graph(self):
        if False:
            print('Hello World!')
        (images1, labels1) = self.load()
        (images2, labels2) = self.load()
        return (images1, images2)

class SinkTestPipeline(CommonPipeline):

    def __init__(self, batch_size, device_id, seed, image_dir, function):
        if False:
            return 10
        super().__init__(batch_size, 1, device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(function=function, num_outputs=0)

    def define_graph(self):
        if False:
            for i in range(10):
                print('nop')
        (images, labels) = self.load()
        self.python_function(images)
        return images

class PythonOperatorInputSetsPipeline(PythonOperatorPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(batch_size, num_threads, device_id, seed, image_dir, function)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        if False:
            return 10
        (images, labels) = self.load()
        processed = self.python_function([images, images])
        return processed

def random_seed():
    if False:
        i = 10
        return i + 15
    return int(random.random() * (1 << 32))
DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 16
SEED = random_seed()
NUM_WORKERS = 6

def run_case(func):
    if False:
        return 10
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func)
    pipe.build()
    pyfunc_pipe.build()
    for it in range(ITERS):
        (preprocessed_output,) = pipe.run()
        (output,) = pyfunc_pipe.run()
        for i in range(len(output)):
            assert numpy.array_equal(output.at(i), func(preprocessed_output.at(i)))

def one_channel_normalize(image):
    if False:
        i = 10
        return i + 15
    return image[:, :, 1] / 255.0

def channels_mean(image):
    if False:
        for i in range(10):
            print('nop')
    r = numpy.mean(image[:, :, 0])
    g = numpy.mean(image[:, :, 1])
    b = numpy.mean(image[:, :, 2])
    return numpy.array([r, g, b])

def bias(image):
    if False:
        i = 10
        return i + 15
    return numpy.array(image > 127, dtype=bool)

def flip(image):
    if False:
        for i in range(10):
            print('nop')
    return numpy.fliplr(image)

def flip_batch(images):
    if False:
        i = 10
        return i + 15
    return [flip(x) for x in images]

def dlflip(image):
    if False:
        i = 10
        return i + 15
    image = ops._dlpack_to_array(image)
    out = numpy.fliplr(image)
    out = ops._dlpack_from_array(out)
    return out

def dlflip_batch(images):
    if False:
        i = 10
        return i + 15
    return [dlflip(x) for x in images]

def Rotate(image):
    if False:
        return 10
    return numpy.rot90(image)

def Brightness(image):
    if False:
        print('Hello World!')
    return numpy.array(ImageEnhance.Brightness(Image.fromarray(image)).enhance(0.5))

def test_python_operator_one_channel_normalize():
    if False:
        print('Hello World!')
    run_case(one_channel_normalize)

def test_python_operator_channels_mean():
    if False:
        i = 10
        return i + 15
    run_case(channels_mean)

def test_python_operator_bias():
    if False:
        while True:
            i = 10
    run_case(bias)

def test_python_operator_flip():
    if False:
        return 10
    dali_flip = FlippingPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_flip = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, flip)
    dali_flip.build()
    numpy_flip.build()
    for it in range(ITERS):
        (numpy_output,) = numpy_flip.run()
        (dali_output,) = dali_flip.run()
        for i in range(len(numpy_output)):
            assert numpy.array_equal(numpy_output.at(i), dali_output.at(i))

class RotatePipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        if False:
            while True:
                i = 10
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.rotate = ops.Rotate(angle=90.0, interp_type=types.INTERP_NN)

    def define_graph(self):
        if False:
            for i in range(10):
                print('nop')
        (images, labels) = self.load()
        rotate = self.rotate(images)
        return rotate

class BrightnessPipeline(CommonPipeline):

    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        if False:
            return 10
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.brightness = ops.BrightnessContrast(device='gpu', brightness=0.5)

    def define_graph(self):
        if False:
            i = 10
            return i + 15
        (images, labels) = self.load()
        bright = self.brightness(images.gpu())
        return bright

def test_python_operator_rotate():
    if False:
        print('Hello World!')
    dali_rotate = RotatePipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_rotate = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, Rotate)
    dali_rotate.build()
    numpy_rotate.build()
    for it in range(ITERS):
        (numpy_output,) = numpy_rotate.run()
        (dali_output,) = dali_rotate.run()
        for i in range(len(numpy_output)):
            if not numpy.array_equal(numpy_output.at(i), dali_output.at(i)):
                cv2.imwrite('numpy.png', numpy_output.at(i))
                cv2.imwrite('dali.png', dali_output.at(i))
                assert numpy.array_equal(numpy_output.at(i), dali_output.at(i))

def test_python_operator_brightness():
    if False:
        print('Hello World!')
    dali_brightness = BrightnessPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_brightness = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, Brightness)
    dali_brightness.build()
    numpy_brightness.build()
    for it in range(ITERS):
        (numpy_output,) = numpy_brightness.run()
        (dali_output,) = dali_brightness.run()
        for i in range(len(dali_output)):
            assert numpy.allclose(numpy_output.at(i), dali_output.as_cpu().at(i), rtol=1e-05, atol=1)

def invalid_function(image):
    if False:
        print('Hello World!')
    return img

@raises(RuntimeError, 'img*not defined')
def test_python_operator_invalid_function():
    if False:
        for i in range(10):
            print('nop')
    invalid_pipe = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, invalid_function)
    invalid_pipe.build()
    invalid_pipe.run()

@raises(TypeError, 'do not support multiple input sets')
def test_python_operator_with_input_sets():
    if False:
        while True:
            i = 10
    invalid_pipe = PythonOperatorInputSetsPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, Rotate)
    invalid_pipe.build()

def split_red_blue(image):
    if False:
        for i in range(10):
            print('nop')
    return (image[:, :, 0], image[:, :, 2])

def mixed_types(image):
    if False:
        i = 10
        return i + 15
    return (bias(image), one_channel_normalize(image))

def run_two_outputs(func):
    if False:
        while True:
            i = 10
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func)
    pipe.build()
    pyfunc_pipe.build()
    for it in range(ITERS):
        (preprocessed_output,) = pipe.run()
        (output1, output2) = pyfunc_pipe.run()
        for i in range(len(output1)):
            (pro1, pro2) = func(preprocessed_output.at(i))
            assert numpy.array_equal(output1.at(i), pro1)
            assert numpy.array_equal(output2.at(i), pro2)

def test_split():
    if False:
        i = 10
        return i + 15
    run_two_outputs(split_red_blue)

def test_mixed_types():
    if False:
        i = 10
        return i + 15
    run_two_outputs(mixed_types)

def multi_per_sample_compare(func, pipe, pyfunc_pipe):
    if False:
        print('Hello World!')
    for it in range(ITERS):
        (preprocessed_output1, preprocessed_output2) = pipe.run()
        (out1, out2, out3) = pyfunc_pipe.run()
        for i in range(BATCH_SIZE):
            (pro1, pro2, pro3) = func(preprocessed_output1.at(i), preprocessed_output2.at(i))
            assert numpy.array_equal(out1.at(i), pro1)
            assert numpy.array_equal(out2.at(i), pro2)
            assert numpy.array_equal(out3.at(i), pro3)

def multi_batch_compare(func, pipe, pyfunc_pipe):
    if False:
        print('Hello World!')
    for it in range(ITERS):
        (preprocessed_output1, preprocessed_output2) = pipe.run()
        (out1, out2, out3) = pyfunc_pipe.run()
        in1 = [preprocessed_output1.at(i) for i in range(BATCH_SIZE)]
        in2 = [preprocessed_output2.at(i) for i in range(BATCH_SIZE)]
        (pro1, pro2, pro3) = func(in1, in2)
        for i in range(BATCH_SIZE):
            assert numpy.array_equal(out1.at(i), pro1[i])
            assert numpy.array_equal(out2.at(i), pro2[i])
            assert numpy.array_equal(out3.at(i), pro3[i])

def run_multi_input_multi_output(func, compare, batch=False):
    if False:
        while True:
            i = 10
    pipe = DoubleLoadPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = MultiInputMultiOutputPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func, batch_processing=batch)
    pipe.build()
    pyfunc_pipe.build()
    compare(func, pipe, pyfunc_pipe)

def split_and_mix(images1, images2):
    if False:
        while True:
            i = 10
    r = (images1[:, :, 0] + images2[:, :, 0]) // 2
    g = (images1[:, :, 1] + images2[:, :, 1]) // 2
    b = (images1[:, :, 2] + images2[:, :, 2]) // 2
    return (r, g, b)

def output_with_stride_mixed_types(images1, images2):
    if False:
        i = 10
        return i + 15
    return (images1[:, :, 2], one_channel_normalize(images2), images1 > images2)

def test_split_and_mix():
    if False:
        for i in range(10):
            print('nop')
    run_multi_input_multi_output(split_and_mix, multi_per_sample_compare)

def test_output_with_stride_mixed_types():
    if False:
        return 10
    run_multi_input_multi_output(output_with_stride_mixed_types, multi_per_sample_compare)

def mix_and_split_batch(images1, images2):
    if False:
        i = 10
        return i + 15
    mixed = [(images1[i] + images2[i]) // 2 for i in range(len(images1))]
    r = [im[:, :, 0] for im in mixed]
    g = [im[:, :, 1] for im in mixed]
    b = [im[:, :, 2] for im in mixed]
    return (r, g, b)

def with_stride_mixed_types_batch(images1, images2):
    if False:
        print('Hello World!')
    out1 = [im[:, :, 2] for im in images1]
    out2 = [one_channel_normalize(im) for im in images2]
    out3 = [im1 > im2 for (im1, im2) in zip(images1, images2)]
    return (out1, out2, out3)

def test_split_and_mix_batch():
    if False:
        for i in range(10):
            print('nop')
    run_multi_input_multi_output(mix_and_split_batch, multi_batch_compare, batch=True)

def test_output_with_stride_mixed_types_batch():
    if False:
        print('Hello World!')
    run_multi_input_multi_output(with_stride_mixed_types_batch, multi_batch_compare, batch=True)

@raises(Exception, 'must be a tuple')
def test_not_a_tuple():
    if False:
        for i in range(10):
            print('nop')
    invalid_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, flip_batch)
    invalid_pipe.build()
    invalid_pipe.run()

@raises(Exception, 'must be a tuple')
def test_not_a_tuple_dl():
    if False:
        while True:
            i = 10
    invalid_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, dlflip_batch, op=ops.DLTensorPythonFunction)
    invalid_pipe.build()
    invalid_pipe.run()

def three_outputs(inp):
    if False:
        for i in range(10):
            print('nop')
    return (inp, inp, inp)

@raises(Exception, glob='Unexpected number of outputs*got 3*expected 2')
def test_wrong_outputs_number():
    if False:
        while True:
            i = 10
    invalid_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, three_outputs)
    invalid_pipe.build()
    invalid_pipe.run()

@raises(Exception, glob='Unexpected number of outputs*got 3*expected 2')
def test_wrong_outputs_number_dl():
    if False:
        while True:
            i = 10
    invalid_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, three_outputs, op=ops.DLTensorPythonFunction)
    invalid_pipe.build()
    invalid_pipe.run()
SINK_PATH = tempfile.mkdtemp()

def save(image):
    if False:
        print('Hello World!')
    Image.fromarray(image).save(SINK_PATH + '/sink_img' + str(time.process_time()) + '.jpg', 'JPEG')

def test_sink():
    if False:
        print('Hello World!')
    pipe = SinkTestPipeline(BATCH_SIZE, DEVICE_ID, SEED, images_dir, save)
    pipe.build()
    if not os.path.exists(SINK_PATH):
        os.mkdir(SINK_PATH)
    assert len(glob.glob(SINK_PATH + '/sink_img*')) == 0
    pipe.run()
    created_files = glob.glob(SINK_PATH + '/sink_img*')
    print(created_files)
    assert len(created_files) == BATCH_SIZE
    for file in created_files:
        os.remove(file)
    os.rmdir(SINK_PATH)
counter = 0

def func_with_side_effects(images):
    if False:
        while True:
            i = 10
    global counter
    counter = counter + 1
    return numpy.full_like(images, counter)

def test_func_with_side_effects():
    if False:
        i = 10
        return i + 15
    pipe_one = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func_with_side_effects, prefetch_queue_depth=1)
    pipe_two = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func_with_side_effects, prefetch_queue_depth=1)
    pipe_one.build()
    pipe_two.build()
    global counter
    for it in range(ITERS):
        counter = 0
        (out_one,) = pipe_one.run()
        (out_two,) = pipe_two.run()
        assert counter == len(out_one) + len(out_two)
        elems_one = [out_one.at(s)[0][0][0] for s in range(BATCH_SIZE)]
        elems_one.sort()
        assert elems_one == [i for i in range(1, BATCH_SIZE + 1)]
        elems_two = [out_two.at(s)[0][0][0] for s in range(BATCH_SIZE)]
        elems_two.sort()
        assert elems_two == [i for i in range(BATCH_SIZE + 1, 2 * BATCH_SIZE + 1)]

class AsyncPipeline(Pipeline):

    def __init__(self, batch_size, num_threads, device_id, _seed):
        if False:
            print('Hello World!')
        super().__init__(batch_size, num_threads, device_id, seed=_seed, exec_async=True, exec_pipelined=True)
        self.op = ops.PythonFunction(function=lambda : numpy.zeros([2, 2, 2]))

    def define_graph(self):
        if False:
            while True:
                i = 10
        return self.op()

def test_output_layout():
    if False:
        for i in range(10):
            print('nop')
    pipe = CommonPipeline(1, 1, 0, 999, images_dir)
    with pipe:
        (images, _) = pipe.load()
        (out1, out2) = fn.python_function(images, function=lambda x: (x, x.mean(2)), num_outputs=2, output_layouts=['ABC', 'DE'])
        (out3, out4) = fn.python_function(images, function=lambda x: (x, x / 2), num_outputs=2, output_layouts='FGH')
        (out5, out6) = fn.python_function(images, function=lambda x: (x, x / 2), num_outputs=2, output_layouts=['IJK'])
        pipe.set_outputs(out1, out2, out3, out4, out5, out6)
    pipe.build()
    (out1, out2, out3, out4, out5, out6) = pipe.run()
    assert out1.layout() == 'ABC'
    assert out2.layout() == 'DE'
    assert out3.layout() == 'FGH'
    assert out4.layout() == 'FGH'
    assert out5.layout() == 'IJK'
    assert out6.layout() == ''

@raises(RuntimeError, '*length of*output_layouts*greater than*')
def test_invalid_layouts_arg():
    if False:
        print('Hello World!')
    pipe = Pipeline(1, 1, 0, 999, exec_async=False, exec_pipelined=False)
    with pipe:
        out = fn.python_function(function=lambda : numpy.zeros((1, 1)), output_layouts=['HW', 'HWC'])
        pipe.set_outputs(out)
    pipe.build()
    pipe.run()

def test_python_function_conditionals():
    if False:
        while True:
            i = 10
    batch_size = 32

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=4, exec_async=False, exec_pipelined=False, enable_conditionals=True)
    def py_fun_pipeline():
        if False:
            i = 10
            return i + 15
        predicate = fn.external_source(source=lambda sample_info: numpy.array(sample_info.idx_in_batch < batch_size / 2), batch=False)
        if predicate:
            (out1, out2) = fn.python_function(predicate, num_outputs=2, function=lambda _: (numpy.array(42), numpy.array(10)))
        else:
            out1 = fn.python_function(function=lambda : numpy.array(0))
            out2 = types.Constant(numpy.array(0), device='cpu', dtype=types.INT64)
        return (out1, out2)
    pipe = py_fun_pipeline()
    pipe.build()
    pipe.run()