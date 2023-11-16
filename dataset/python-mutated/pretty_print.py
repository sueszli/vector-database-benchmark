def max_array_length(arr_max, arr_to_compare):
    if False:
        print('Hello World!')
    for i in range(len(arr_max)):
        str_length = len(arr_to_compare[i])
        if arr_max[i] < str_length:
            arr_max[i] = str_length
    return arr_max

def get_string(table_array, max_arr):
    if False:
        while True:
            i = 10
    temp_str = ''
    for row in table_array:
        temp_str += '\n'
        for col_no in range(len(row)):
            max_col = max_arr[col_no]
            length = len(row[col_no])
            starting_loc = (max_col - length) // 2
            temp_str += ' ' * starting_loc + row[col_no] + ' ' * (max_col - length - starting_loc)
    return temp_str

def summary_tensor(tensor):
    if False:
        print('Hello World!')
    head = ['htype', 'shape', 'dtype', 'compression']
    divider = ['-------'] * 4
    max_column_length = [7, 7, 7, 7]
    tensor_htype = tensor.htype
    if tensor_htype == None:
        tensor_htype = 'None'
    tensor_shape = str(tensor.shape_interval)
    tensor_compression = tensor.meta.sample_compression
    if tensor_compression == None:
        tensor_compression = 'None'
    if tensor.dtype == None:
        tensor_dtype = 'None'
    else:
        tensor_dtype = tensor.dtype.name
    self_array = [head, divider, [tensor_htype, tensor_shape, tensor_dtype, tensor_compression]]
    max_column_length = max_array_length(max_column_length, self_array[2])
    max_column_length = [elem + 2 for elem in max_column_length]
    return get_string(self_array, max_column_length)

def summary_dataset(dataset):
    if False:
        i = 10
        return i + 15
    head = ['tensor', 'htype', 'shape', 'dtype', 'compression']
    divider = ['-------'] * 5
    tensor_dict = dataset.tensors
    max_column_length = [7, 7, 7, 7, 7]
    count = 0
    table_array = [head, divider]
    for tensor_name in tensor_dict:
        tensor_object = tensor_dict[tensor_name]
        tensor_htype = tensor_object.htype
        if tensor_htype == None:
            tensor_htype = 'None'
        tensor_shape = str(tensor_object.shape_interval)
        tensor_compression = tensor_object.meta.sample_compression
        if tensor_compression == None:
            tensor_compression = 'None'
        if tensor_object.dtype == None:
            tensor_dtype = 'None'
        else:
            tensor_dtype = tensor_object.dtype.name
        row_array = [tensor_name, tensor_htype, tensor_shape, tensor_dtype, tensor_compression]
        table_array.append(row_array)
        max_column_length = max_array_length(max_column_length, row_array)
        count += 1
    max_column_length = [elem + 2 for elem in max_column_length]
    return get_string(table_array, max_column_length)