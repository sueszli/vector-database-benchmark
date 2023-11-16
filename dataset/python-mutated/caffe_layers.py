convolutionDefinition = '\nname : "ConvolutionTest"\ninput : "data"\ninput_shape {dim:1 dim :3 dim :5 dim :5}\nlayer {\n name : "convolution"\n type : "Convolution"\n bottom : "data"\n top : "convolution"\n convolution_param {\n  num_output : 4\n  kernel_size: 2\n  weight_filler {\n   type: "xavier"\n }\n  bias_filler {\n  type: "gaussian"\n  std: 0.02\n   }\n  }\n }\n'
convolutionShapes = [{'data': (1, 3, 5, 5)}]
convolutionName = 'convolution'
reluDefinition = '\nname : "ReluTest"\ninput : "data"\ninput_shape{dim:2 dim :2}\n layer {\n  name: "relu"\n  type: "ReLU"\n  bottom: "data"\n  top: "relu"\n}\n'
reluShapes = [{'data': (2, 2)}]
reluName = 'relu'
crossMapLrnDefinition = '\nname : "SpatialCrossMapLRNTest"\ninput : "data"\ninput_shape{dim:1 dim :3 dim:224 dim :224}\nlayer {\n  name: "crossMapLrn"\n  type: "LRN"\n  bottom: "data"\n  top: "crossMapLrn"\n  lrn_param {\n    local_size: 5\n    alpha: 1.0E-4\n    beta: 0.75\n    k: 1.0\n  }\n}\n'
crossMapLrnShapes = [{'data': (1, 3, 224, 224)}]
crossMapLrnName = 'crossMapLrn'
withinChannelLRNDefinition = '\nname : "SpatialWithinChannelLRNTest"\ninput : "data"\ninput_shape{dim:1 dim :3 dim:224 dim :224}\nlayer {\n  name: "withinChannelLRN"\n  type: "LRN"\n  bottom: "data"\n  top: "withinChannelLRN"\n  lrn_param {\n    local_size: 5\n    alpha: 1.0E-4\n    beta: 0.75\n    k: 1.0\n    norm_region : WITHIN_CHANNEL\n  }\n}\n'
withinChannelLRNShapes = [{'data': (1, 3, 224, 224)}]
withinChannelLRNName = 'withinChannelLRN'
innerProductDefinition = '\nname : "InnerProductTest"\ninput : "data"\ninput_shape{dim: 2 dim: 10}\nlayer {\n  name: "innerProduct"\n  type: "InnerProduct"\n  bottom: "data"\n  top: "innerProduct"\n  inner_product_param {\n    num_output: 10\n  }\n}\n'
innerProductShapes = [{'data': (2, 10)}]
innerProductName = 'innerProduct'
maxpoolingDefinition = '\nname : "MaxpoolingTest"\ninput : "data"\ninput_shape{dim: 1 dim: 3 dim: 3 dim: 3}\nlayer {\n  name: "maxpooling"\n  type: "Pooling"\n  bottom: "data"\n  top: "maxpooling"\n  pooling_param {\n    pool: MAX\n    kernel_size: 2\n    stride: 2\n  }\n}\n'
maxpoolingShapes = [{'data': (1, 3, 3, 3)}]
maxpoolingName = 'maxpooling'
avepoolingDefinition = '\nname : "AvepoolingTest"\ninput : "data"\ninput_shape{dim: 1 dim: 3 dim: 3 dim: 3}\nlayer {\n  name: "avepooling"\n  type: "Pooling"\n  bottom: "data"\n  top: "avepooling"\n  pooling_param {\n    pool: AVE\n    kernel_size: 2\n    stride: 2\n  }\n}\n'
avepoolingShapes = [{'data': (1, 3, 3, 3)}]
avepoolingName = 'avepooling'
softMaxDefinition = '\nname : "SoftMaxTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "softMax"\n  type: "Softmax"\n  bottom: "data"\n  top: "softMax"\n}\n'
softMaxShapes = [{'data': (2, 2)}]
softMaxName = 'softMax'
tanhDefinition = '\nname : "TanhTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "tanh"\n  type: "TanH"\n  bottom: "data"\n  top: "tanh"\n}\n'
tanhShapes = [{'data': (2, 2)}]
tanhName = 'tanh'
sigmoidDefinition = '\nname : "SigmoidTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "sigmoid"\n  type: "Sigmoid"\n  bottom: "data"\n  top: "sigmoid"\n}\n'
sigmoidShapes = [{'data': (2, 2)}]
sigmoidName = 'sigmoid'
absDefinition = '\nname : "AbsTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "abs"\n  type: "AbsVal"\n  bottom: "data"\n  top: "abs"\n}\n'
absShapes = [{'data': (2, 2)}]
absName = 'abs'
batchNormDefinition = '\nname : "BatchNormTest"\ninput: "data"\ninput_dim: 1\ninput_dim: 3\ninput_dim: 224\ninput_dim: 224\n\nlayer {\n        bottom: "data"\n        top: "conv1"\n        name: "conv1"\n        type: "Convolution"\n        convolution_param {\n                num_output: 64\n                kernel_size: 7\n                pad: 3\n                stride: 2\n        }\n}\n\nlayer {\n        bottom: "conv1"\n        top: "batchNorm"\n        name: "batchNorm"\n        type: "BatchNorm"\n        batch_norm_param {\n                use_global_stats: true\n        }\n}\n'
batchNormShapes = [{'data': (1, 3, 224, 224)}]
batchNormName = 'batchNorm'
concatDefinition = '\nname : "ConcatTest"\ninput : "data1"\ninput_shape{dim: 2 dim: 2}\ninput : "data2"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "abs"\n  type: "AbsVal"\n  bottom: "data1"\n  top: "abs"\n}\nlayer {\n  name: "sigmoid"\n  type: "Sigmoid"\n  bottom: "data2"\n  top: "sigmoid"\n}\nlayer {\n  name: "concat"\n  type: "Concat"\n  bottom: "abs"\n  bottom: "sigmoid"\n  top: "concat"\n}\n'
concatShapes = [{'data1': (2, 2)}, {'data2': (2, 2)}]
concatName = 'concat'
eluDefinition = '\nname : "EluTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "elu"\n  type: "ELU"\n  bottom: "data"\n  top: "elu"\n}\n'
eluShapes = [{'data': (2, 2)}]
eluName = 'elu'
flattenDefinition = '\nname : "FlattenTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "flatten"\n  type: "Flatten"\n  bottom: "data"\n  top: "flatten"\n}\n'
flattenShapes = [{'data': (2, 2)}]
flattenName = 'flatten'
logDefinition = '\nname : "LogTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "log"\n  type: "Log"\n  bottom: "data"\n  top: "log"\n}\n'
logShapes = [{'data': (2, 2)}]
logName = 'log'
powerDefinition = '\nname : "PowerTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "power"\n  type: "Power"\n  bottom: "data"\n  top: "power"\n}\n'
powerShapes = [{'data': (2, 2)}]
powerName = 'power'
preluDefinition = '\nname : "PReLUTest"\ninput : "data"\ninput_shape{dim: 2 dim: 5}\nlayer {\n  name: "prelu"\n  type: "PReLU"\n  bottom: "data"\n  top: "prelu"\n}\n'
preluShapes = [{'data': (2, 5)}]
preluName = 'prelu'
reshapeDefinition = '\nname : "ReshapeTest"\ninput : "data"\ninput_shape{dim: 2 dim: 8}\nlayer {\n  name: "reshape"\n  type: "Reshape"\n  bottom: "data"\n  top: "reshape"\n  reshape_param { shape { dim:  0 dim:  -1  dim:  4 } }\n}\n'
reshapeShapes = [{'data': (2, 8)}]
reshapeName = 'reshape'
scaleDefinition = '\nname : "ScaleTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "scale"\n  type: "Scale"\n  bottom: "data"\n  top: "scale"\n}\n'
scaleShapes = [{'data': (2, 2)}]
scaleName = 'scale'
biasDefinition = '\nname : "BiasTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "bias"\n  type: "Bias"\n  bottom: "data"\n  top: "bias"\n}\n'
biasShapes = [{'data': (2, 2)}]
biasName = 'bias'
thresholdDefinition = '\nname : "ThresholdTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "threshold"\n  type: "Threshold"\n  bottom: "data"\n  top: "threshold"\n  threshold_param {\n    threshold : 0.5\n  }\n}\n'
thresholdShapes = [{'data': (2, 2)}]
thresholdName = 'threshold'
expDefinition = '\nname : "ExpTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "exp"\n  type: "Exp"\n  bottom: "data"\n  top: "exp"\n}\n'
expShapes = [{'data': (2, 2)}]
expName = 'exp'
sliceDefinition = '\nname : "SliceTest"\ninput : "data"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "slice"\n  type: "Slice"\n  bottom: "data"\n  top: "slice"\n}\n'
sliceShapes = [{'data': (2, 2)}]
sliceName = 'slice'
tileDefinition = '\nname : "TileTest"\ninput : "data"\ninput_shape{dim: 2 dim : 2}\nlayer {\n  name: "tile"\n  type: "Tile"\n  bottom: "data"\n  top: "tile"\n  tile_param {\n    axis : 1\n    tiles : 2\n  }\n}\n'
tileShapes = [{'data': (2, 2)}]
tileName = 'tile'
eltwiseMaxDefinition = '\nname : "EltwiseMaxTest"\ninput : "data1"\ninput_shape{dim: 2 dim: 2}\ninput : "data2"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "abs"\n  type: "AbsVal"\n  bottom: "data1"\n  top: "abs"\n}\nlayer {\n  name: "sigmoid"\n  type: "Sigmoid"\n  bottom: "data2"\n  top: "sigmoid"\n}\nlayer {\n  name: "eltwiseMax"\n  type: "Eltwise"\n  bottom: "abs"\n  bottom: "sigmoid"\n  top: "eltwiseMax"\n  eltwise_param {\n    operation : MAX\n  }\n}\n'
eltwiseMaxShapes = [{'data1': (2, 2)}, {'data2': (2, 2)}]
eltwiseMaxName = 'eltwiseMax'
eltwiseProdDefinition = '\nname : "EltwiseProdTest"\ninput : "data1"\ninput_shape{dim: 2 dim: 2}\ninput : "data2"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "abs"\n  type: "AbsVal"\n  bottom: "data1"\n  top: "abs"\n}\nlayer {\n  name: "sigmoid"\n  type: "Sigmoid"\n  bottom: "data2"\n  top: "sigmoid"\n}\nlayer {\n  name: "eltwiseProd"\n  type: "Eltwise"\n  bottom: "abs"\n  bottom: "sigmoid"\n  top: "eltwiseProd"\n  eltwise_param {\n    operation : PROD\n  }\n}\n'
eltwiseProdShapes = [{'data1': (2, 2)}, {'data2': (2, 2)}]
eltwiseProdName = 'eltwiseProd'
eltwiseSUMDefinition = '\nname : "EltwiseSUMTest"\ninput : "data1"\ninput_shape{dim: 2 dim: 2}\ninput : "data2"\ninput_shape{dim: 2 dim: 2}\nlayer {\n  name: "abs1"\n  type: "AbsVal"\n  bottom: "data1"\n  top: "abs1"\n}\nlayer {\n  name: "abs2"\n  type: "AbsVal"\n  bottom: "data2"\n  top: "abs2"\n}\nlayer {\n  name: "eltwiseSUM"\n  type: "Eltwise"\n  bottom: "abs1"\n  bottom: "abs2"\n  top: "eltwiseSUM"\n  eltwise_param {\n    operation : SUM\n     coeff: [0.5 , 1.0]\n  }\n}\n'
eltwiseSUMShapes = [{'data1': (2, 2)}, {'data2': (2, 2)}]
eltwiseSUMName = 'eltwiseSUM'
deconvolutionDefinition = '\nname : "deconvolution"\ninput : "data"\ninput_shape {dim:1 dim :3 dim :5 dim :5}\nlayer {\n  name: "deconvolution"\n  type: "Deconvolution"\n  bottom: "data"\n  top: "deconvolution"\n  convolution_param {\n    num_output: 4\n    pad: 0\n    kernel_size: 2\n    stride: 2\n    weight_filler {\n      type: "xavier"\n    }\n  }\n}\n\n'
deconvolutionShapes = [{'data': (1, 3, 5, 5)}]
deconvolutionName = 'deconvolution'
testlayers = []

class caffe_test_layer:

    def __init__(self, name, definition, shapes):
        if False:
            while True:
                i = 10
        self.name = name
        self.definition = definition
        self.shapes = shapes

def registerTestLayer(name, definition, shapes):
    if False:
        while True:
            i = 10
    layer = caffe_test_layer(name, definition, shapes)
    testlayers.append(layer)
registerTestLayer(convolutionName, convolutionDefinition, convolutionShapes)
registerTestLayer(reluName, reluDefinition, reluShapes)
registerTestLayer(crossMapLrnName, crossMapLrnDefinition, crossMapLrnShapes)
registerTestLayer(withinChannelLRNName, withinChannelLRNDefinition, withinChannelLRNShapes)
registerTestLayer(innerProductName, innerProductDefinition, innerProductShapes)
registerTestLayer(maxpoolingName, maxpoolingDefinition, maxpoolingShapes)
registerTestLayer(avepoolingName, avepoolingDefinition, avepoolingShapes)
registerTestLayer(softMaxName, softMaxDefinition, softMaxShapes)
registerTestLayer(tanhName, tanhDefinition, tanhShapes)
registerTestLayer(sigmoidName, sigmoidDefinition, sigmoidShapes)
registerTestLayer(absName, absDefinition, absShapes)
registerTestLayer(batchNormName, batchNormDefinition, batchNormShapes)
registerTestLayer(concatName, concatDefinition, concatShapes)
registerTestLayer(eluName, eluDefinition, eluShapes)
registerTestLayer(flattenName, flattenDefinition, flattenShapes)
registerTestLayer(logName, logDefinition, logShapes)
registerTestLayer(powerName, powerDefinition, powerShapes)
registerTestLayer(preluName, preluDefinition, preluShapes)
registerTestLayer(reshapeName, reshapeDefinition, reshapeShapes)
registerTestLayer(scaleName, scaleDefinition, scaleShapes)
registerTestLayer(biasName, biasDefinition, biasShapes)
registerTestLayer(thresholdName, thresholdDefinition, thresholdShapes)
registerTestLayer(expName, expDefinition, expShapes)
registerTestLayer(sliceName, sliceDefinition, sliceShapes)
registerTestLayer(tileName, tileDefinition, tileShapes)
registerTestLayer(eltwiseMaxName, eltwiseMaxDefinition, eltwiseMaxShapes)
registerTestLayer(eltwiseProdName, eltwiseProdDefinition, eltwiseProdShapes)
registerTestLayer(eltwiseSUMName, eltwiseSUMDefinition, eltwiseSUMShapes)
registerTestLayer(deconvolutionName, deconvolutionDefinition, deconvolutionShapes)