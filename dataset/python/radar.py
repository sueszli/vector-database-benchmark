from ... import options as opts
from ... import types
from ...charts.chart import Chart
from ...globals import ChartType


class Radar(Chart):
    """
    <<< Radar >>>

    Radar maps are mainly used to represent multivariable data.
    """

    def add_schema(
        self,
        schema: types.Sequence[types.Union[opts.RadarIndicatorItem, dict]],
        shape: types.Optional[str] = None,
        center: types.Optional[types.Sequence] = None,
        radius: types.Optional[types.Union[types.Sequence, str]] = None,
        start_angle: types.Numeric = 90,
        textstyle_opts: types.TextStyle = opts.TextStyleOpts(),
        splitline_opt: types.SplitLine = opts.SplitLineOpts(is_show=True),
        splitarea_opt: types.SplitArea = opts.SplitAreaOpts(),
        axisline_opt: types.AxisLine = opts.AxisLineOpts(),
        radiusaxis_opts: types.RadiusAxis = None,
        angleaxis_opts: types.AngleAxis = None,
        polar_opts: types.Polar = None,
    ):
        self.options.update(
            radiusAxis=radiusaxis_opts, angleAxis=angleaxis_opts, polar=polar_opts
        )

        indicators = []
        for s in schema:
            if isinstance(s, opts.RadarIndicatorItem):
                s = s.opts
            indicators.append(s)

        if self.options.get("radar") is None:
            self.options.update(radar=[])

        self.options.get("radar").append(
            {
                "indicator": indicators,
                "shape": shape,
                "center": center,
                "radius": radius,
                "startAngle": start_angle,
                "name": {"textStyle": textstyle_opts},
                "splitLine": splitline_opt,
                "splitArea": splitarea_opt,
                "axisLine": axisline_opt,
            }
        )
        return self

    def add(
        self,
        series_name: str,
        data: types.Sequence[types.Union[opts.RadarItem, dict]],
        *,
        symbol: types.Optional[str] = None,
        color: types.Optional[str] = None,
        label_opts: opts.LabelOpts = opts.LabelOpts(),
        radar_index: types.Numeric = None,
        linestyle_opts: opts.LineStyleOpts = opts.LineStyleOpts(),
        areastyle_opts: opts.AreaStyleOpts = opts.AreaStyleOpts(),
        tooltip_opts: types.Tooltip = None,
    ):
        if all([isinstance(d, opts.RadarItem) for d in data]):
            for a in data:
                self._append_legend(a.get("name"))
        else:
            self._append_legend(series_name)
        self.options.get("series").append(
            {
                "type": ChartType.RADAR,
                "name": series_name,
                "data": data,
                "symbol": symbol,
                "label": label_opts,
                "radarIndex": radar_index,
                "itemStyle": {"normal": {"color": color}},
                "lineStyle": linestyle_opts,
                "areaStyle": areastyle_opts,
                "tooltip": tooltip_opts,
            }
        )
        return self
