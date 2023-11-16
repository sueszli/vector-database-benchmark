from .. import optional_features
try:
    from nicegui_highcharts import highchart
    optional_features.register('highcharts')
    __all__ = ['highchart']
except ImportError:

    class highchart:

        def __init__(self, *args, **kwargs) -> None:
            if False:
                while True:
                    i = 10
            raise NotImplementedError('Highcharts is not installed. Please run `pip install nicegui[highcharts]`.')