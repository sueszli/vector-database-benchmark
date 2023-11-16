def preprocess_dummy_data(rank, data):
    if False:
        return 10
    '\n    A function that moves the data from CPU to GPU\n    for DummyData class.\n    Args:\n        rank (int): worker rank\n        data (list): training examples\n    '
    for i in range(len(data)):
        data[i][0] = data[i][0].cuda(rank)
        data[i][1] = data[i][1].cuda(rank)
    return data