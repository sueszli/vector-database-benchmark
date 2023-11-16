from tensorflow.keras import layers

def get_global_average_pooling(shape):
    if False:
        while True:
            i = 10
    return [layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D, layers.GlobalAveragePooling3D][len(shape) - 3]

def get_global_max_pooling(shape):
    if False:
        while True:
            i = 10
    return [layers.GlobalMaxPool1D, layers.GlobalMaxPool2D, layers.GlobalMaxPool3D][len(shape) - 3]

def get_max_pooling(shape):
    if False:
        i = 10
        return i + 15
    return [layers.MaxPool1D, layers.MaxPool2D, layers.MaxPool3D][len(shape) - 3]

def get_conv(shape):
    if False:
        while True:
            i = 10
    return [layers.Conv1D, layers.Conv2D, layers.Conv3D][len(shape) - 3]

def get_sep_conv(shape):
    if False:
        return 10
    return [layers.SeparableConv1D, layers.SeparableConv2D, layers.Conv3D][len(shape) - 3]