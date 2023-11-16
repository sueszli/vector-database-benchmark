from .position import position

class position_identity(position):
    """
    Do not adjust the position
    """

    @classmethod
    def compute_layer(cls, data, params, layout):
        if False:
            return 10
        return data