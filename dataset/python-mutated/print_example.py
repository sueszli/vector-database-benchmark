def encode_data(data: list):
    if False:
        while True:
            i = 10
    print('Encode data')
    data_map = {'a': 1, 'b': 2, 'c': 3}
    print(f'Data map: {data_map}')
    return [data_map[num] for num in data]

def add_one(data: list):
    if False:
        while True:
            i = 10
    print('Add one')
    return [num + 1 for num in data]

def process_data(data: list):
    if False:
        print('Hello World!')
    print('Process data')
    data = encode_data(data)
    print(f'Encoded data: {data}')
    data = add_one(data)
    print(f'Added one: {data}')
process_data(['a', 'a', 'c'])