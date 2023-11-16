from autokeras.prototype import base_block

class FlexibleBlock(base_block.BaseBlock):

    def _build_wrapper(self, hp, inputs, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass