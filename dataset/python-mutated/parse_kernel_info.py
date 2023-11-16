import paddle

class KernelInfo:

    def __init__(self, op_type):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = op_type
        self.supported_dtypes = set()

    def parse_phi_dtypes(self, registered_info_list, device='GPU'):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(registered_info_list, list)
        assert device in ['CPU', 'GPU']
        for kernel_str in registered_info_list:
            kernel_strs = kernel_str.replace('(', '').replace(')', '').split(',')
            if device in kernel_strs[0]:
                self.supported_dtypes.add(kernel_strs[-1].replace(' ', ''))

    def parse_fluid_dtypes(self, registered_info_list, device='gpu'):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(registered_info_list, list)
        assert device in ['cpu', 'gpu']
        for kernel_str in registered_info_list:
            kernel_strs = kernel_str.split(';')
            if 'place' in kernel_strs[2] and device in kernel_strs[2]:
                assert 'data_type' in kernel_strs[0]
                dtype_str = kernel_strs[0].replace('{data_type[', '')
                dtype_str = dtype_str.replace('::paddle::platform::', '')
                dtype_str = dtype_str.replace(']', '')
                self.supported_dtypes.add(dtype_str)

class KernelRegistryStatistics:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.num_ops_for_dtypes = {'all': 0, 'float32': 0, 'float16': 0, 'bfloat16': 0}

    def update(self, supported_dtypes):
        if False:
            return 10
        for dtype in supported_dtypes:
            if dtype in self.num_ops_for_dtypes.keys():
                self.num_ops_for_dtypes[dtype] += 1
            elif dtype == 'float':
                self.num_ops_for_dtypes['float32'] += 1
        self.num_ops_for_dtypes['all'] += 1

    def __str__(self):
        if False:
            i = 10
            return i + 15
        res = '{ '
        num_floats = int(self.num_ops_for_dtypes['float32'])
        for (dtype, num) in self.num_ops_for_dtypes.items():
            res += f'{dtype}: {num:4d}'
            if dtype in ['float16', 'bfloat16']:
                if num_floats != 0:
                    percent = float(self.num_ops_for_dtypes[dtype]) / float(num_floats)
                    res += f'({percent * 100:.2f}%)'
                else:
                    res += f'({0:.2f}%)'
            res += ' '
        res += '}'
        return res

def parse_paddle_kernels(lib='phi', kernel_type='function', print_detail=False):
    if False:
        i = 10
        return i + 15
    assert lib in ['fluid', 'phi']
    if lib == 'phi':
        assert kernel_type in ['function', 'structure', 'all']
        kernel_infos = paddle.base.core._get_registered_phi_kernels(kernel_type)
    else:
        assert kernel_type in ['fluid', 'phi', 'all']
        kernel_infos = paddle.base.core._get_all_register_op_kernels(kernel_type)
    max_op_type_lengths = 0
    stats = KernelRegistryStatistics()
    kernel_info_dict = {}
    for (key, value) in kernel_infos.items():
        info = KernelInfo(key)
        if lib == 'phi':
            info.parse_phi_dtypes(value, device='GPU')
        else:
            info.parse_fluid_dtypes(value, device='gpu')
        kernel_info_dict[key] = info
        if len(info.op_type) > max_op_type_lengths:
            max_op_type_lengths = len(info.op_type)
        stats.update(info.supported_dtypes)
    if print_detail:
        print('==================== lib={}, kernel_type={} ===================='.format(lib, kernel_type))
        print('{} : {}'.format('op_type'.ljust(max_op_type_lengths + 4), 'supported_dtypes for GPU'))
        for (key, value) in sorted(kernel_info_dict.items()):
            print('{} : {}'.format(value.op_type.ljust(max_op_type_lengths + 4), value.supported_dtypes))
        print('')
    return stats

def main(lib):
    if False:
        while True:
            i = 10
    assert lib in ['fluid', 'phi']
    print_detail = False
    if lib == 'phi':
        phi_function_kernels_stats = parse_paddle_kernels(lib, 'function', print_detail=False)
        phi_structure_kernels_stats = parse_paddle_kernels(lib, 'structure', print_detail=False)
        phi_all_kernels_stats = parse_paddle_kernels(lib, 'all', print_detail=print_detail)
        print('==================================   phi kernels summary   ==================================')
        print(f'phi function  kernels : {phi_function_kernels_stats}')
        print(f'phi structure kernels : {phi_structure_kernels_stats}')
        print(f'phi all       kernels : {phi_all_kernels_stats}')
        print('')
    else:
        fluid_ops_stats = parse_paddle_kernels(lib, 'fluid', print_detail=False)
        phi_ops_stats = parse_paddle_kernels(lib, 'phi', print_detail=False)
        all_ops_stats = parse_paddle_kernels(lib, 'all', print_detail=print_detail)
        print('================================== fluid operators summary ==================================')
        print(f'fluid operators : {fluid_ops_stats}')
        print(f'phi   operators : {phi_ops_stats}')
        print(f'all   operators : {all_ops_stats}')
        print('')
main(lib='fluid')
main(lib='phi')