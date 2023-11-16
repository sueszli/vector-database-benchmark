from __future__ import print_function
import os
import numpy as np
import cntk as C
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs

def create_mb_source(image_height, image_width, num_channels, map_file):
    if False:
        i = 10
        return i + 15
    transforms = [xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')]
    return C.io.MinibatchSource(C.io.ImageDeserializer(map_file, C.io.StreamDefs(features=C.io.StreamDef(field='image', transforms=transforms), labels=C.io.StreamDef(field='label', shape=1000))), randomize=False)

def eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects):
    if False:
        i = 10
        return i + 15
    loaded_model = load_model(model_file)
    node_in_graph = loaded_model.find_by_name(node_name)
    output_nodes = combine([node_in_graph.owner])
    print('Evaluating model for output node %s' % node_name)
    features_si = minibatch_source['features']
    with open(output_file, 'wb') as results_file:
        for i in range(0, num_objects):
            mb = minibatch_source.next_minibatch(1)
            output = output_nodes.eval(mb[features_si])
            out_values = output[0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt='%.6f')
if __name__ == '__main__':
    base_folder = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(base_folder, '..', '..', '..', 'PretrainedModels', 'ResNet18_ImageNet_CNTK.model')
    map_file = os.path.join(base_folder, '..', 'DataSets', 'Grocery', 'test.txt')
    os.chdir(os.path.join(base_folder, '..', 'DataSets', 'Grocery'))
    if not (os.path.exists(model_file) and os.path.exists(map_file)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)
    image_height = 224
    image_width = 224
    num_channels = 3
    minibatch_source = create_mb_source(image_height, image_width, num_channels, map_file)
    node_name = 'z.x'
    output_file = os.path.join(base_folder, 'layerOutput.txt')
    eval_and_write(model_file, node_name, output_file, minibatch_source, num_objects=5)
    print('Done. Wrote output to %s' % output_file)