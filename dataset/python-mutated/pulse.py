"""Pulses are descriptions of waveform envelopes. They can be transmitted by control electronics
to the device.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple, Union
from qiskit.circuit.parameterexpression import ParameterExpression

class Pulse(ABC):
    """The abstract superclass for pulses. Pulses are complex-valued waveform envelopes. The
    modulation phase and frequency are specified separately from ``Pulse``s.
    """
    __slots__ = ('duration', 'name', '_limit_amplitude')
    limit_amplitude = True

    @abstractmethod
    def __init__(self, duration: Union[int, ParameterExpression], name: Optional[str]=None, limit_amplitude: Optional[bool]=None):
        if False:
            print('Hello World!')
        'Abstract base class for pulses\n        Args:\n            duration: Duration of the pulse\n            name: Optional name for the pulse\n            limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1.\n                             The default value of ``None`` causes the flag value to be\n                             derived from :py:attr:`~limit_amplitude` which is ``True``\n                             by default but may be set by the user to disable amplitude\n                             checks globally.\n        '
        if limit_amplitude is None:
            limit_amplitude = self.__class__.limit_amplitude
        self.duration = duration
        self.name = name
        self._limit_amplitude = limit_amplitude

    @property
    def id(self) -> int:
        if False:
            i = 10
            return i + 15
        'Unique identifier for this pulse.'
        return id(self)

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        "Return a dictionary containing the pulse's parameters."
        pass

    def is_parameterized(self) -> bool:
        if False:
            print('Hello World!')
        'Return True iff the instruction is parameterized.'
        raise NotImplementedError

    def draw(self, style: Optional[Dict[str, Any]]=None, backend=None, time_range: Optional[Tuple[int, int]]=None, time_unit: str='dt', show_waveform_info: bool=True, plotter: str='mpl2d', axis: Optional[Any]=None):
        if False:
            print('Hello World!')
        'Plot the interpolated envelope of pulse.\n\n        Args:\n            style: Stylesheet options. This can be dictionary or preset stylesheet classes. See\n                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXStandard`,\n                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXSimple`, and\n                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXDebugging` for details of\n                preset stylesheets.\n            backend (Optional[BaseBackend]): Backend object to play the input pulse program.\n                If provided, the plotter may use to make the visualization hardware aware.\n            time_range: Set horizontal axis limit. Tuple ``(tmin, tmax)``.\n            time_unit: The unit of specified time range either ``dt`` or ``ns``.\n                The unit of ``ns`` is available only when ``backend`` object is provided.\n            show_waveform_info: Show waveform annotations, i.e. name, of waveforms.\n                Set ``True`` to show additional information about waveforms.\n            plotter: Name of plotter API to generate an output image.\n                One of following APIs should be specified::\n\n                    mpl2d: Matplotlib API for 2D image generation.\n                        Matplotlib API to generate 2D image. Charts are placed along y axis with\n                        vertical offset. This API takes matplotlib.axes.Axes as `axis` input.\n\n                `axis` and `style` kwargs may depend on the plotter.\n            axis: Arbitrary object passed to the plotter. If this object is provided,\n                the plotters use a given ``axis`` instead of internally initializing\n                a figure object. This object format depends on the plotter.\n                See plotter argument for details.\n\n        Returns:\n            Visualization output data.\n            The returned data type depends on the ``plotter``.\n            If matplotlib family is specified, this will be a ``matplotlib.pyplot.Figure`` data.\n        '
        from qiskit.visualization import pulse_drawer
        return pulse_drawer(program=self, style=style, backend=backend, time_range=time_range, time_unit=time_unit, show_waveform_info=show_waveform_info, plotter=plotter, axis=axis)

    @abstractmethod
    def __eq__(self, other: 'Pulse') -> bool:
        if False:
            print('Hello World!')
        return isinstance(other, type(self))

    @abstractmethod
    def __hash__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError