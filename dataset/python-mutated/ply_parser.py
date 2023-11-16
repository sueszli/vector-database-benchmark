from deeplake.util.object_3d import ply_parsers
FILE_FORMAT_TO_DECODER_CLASS = {'binary_little_endian': ply_parsers.PlyBinParser, 'binary_big_endian': ply_parsers.PlyBinParser, 'ascii_with_normals': ply_parsers.PlyASCIIWithNormalsParser, 'ascii': ply_parsers.PlyASCIIParser}

class PlyParser:

    def __init__(self, fmt, ret, sample_info, tensor_name=None, index=None, aslist=False):
        if False:
            return 10
        self.ply_parser = FILE_FORMAT_TO_DECODER_CLASS[fmt](ret, sample_info, tensor_name, index, aslist)

    @property
    def data(self):
        if False:
            while True:
                i = 10
        return self.ply_parser.data()

    @property
    def numpy(self):
        if False:
            print('Hello World!')
        return self.ply_parser.numpy()