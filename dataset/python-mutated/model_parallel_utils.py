from math import ceil

def assert_device_map(device_map, num_blocks):
    if False:
        for i in range(10):
            print('nop')
    blocks = list(range(0, num_blocks))
    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]
    if len(duplicate_blocks) != 0:
        raise ValueError('Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These attention blocks were specified more than once: ' + str(duplicate_blocks))
    if len(missing_blocks) != 0:
        raise ValueError('There are attention blocks for this model that are not specified in the device_map. Add these attention blocks to a device on the device_map: ' + str(missing_blocks))
    if len(extra_blocks) != 0:
        raise ValueError('The device_map contains more attention blocks than this model has. Remove these from the device_map:' + str(extra_blocks))

def get_device_map(n_layers, devices):
    if False:
        for i in range(10):
            print('nop')
    'Returns a dictionary of layers distributed evenly across all devices.'
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = [layers[i:i + n_blocks] for i in range(0, n_layers, n_blocks)]
    return dict(zip(devices, layers_list))