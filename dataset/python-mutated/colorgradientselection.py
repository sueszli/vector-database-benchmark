from typing import Any, Tuple
from AnyQt.QtCore import Qt, QSize, QAbstractItemModel, Property
from AnyQt.QtWidgets import QWidget, QSlider, QFormLayout, QComboBox, QStyle, QSizePolicy
from AnyQt.QtCore import Signal
from Orange.widgets.utils import itemmodels, colorpalettes
from Orange.widgets.utils.spinbox import DoubleSpinBox, DBL_MIN, DBL_MAX
from Orange.widgets.utils.intervalslider import IntervalSlider

class ColorGradientSelection(QWidget):
    activated = Signal(int)
    currentIndexChanged = Signal(int)
    thresholdsChanged = Signal(float, float)
    centerChanged = Signal(float)

    def __init__(self, *args, thresholds=(0.0, 1.0), center=None, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        low = round(clip(thresholds[0], 0.0, 1.0), 2)
        high = round(clip(thresholds[1], 0.0, 1.0), 2)
        high = max(low, high)
        (self.__threshold_low, self.__threshold_high) = (low, high)
        self.__center = center
        form = QFormLayout(formAlignment=Qt.AlignLeft, labelAlignment=Qt.AlignLeft, fieldGrowthPolicy=QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(0, 0, 0, 0)
        self.gradient_cb = QComboBox(None, objectName='gradient-combo-box')
        self.gradient_cb.setAttribute(Qt.WA_LayoutUsesWidgetRect)
        icsize = self.style().pixelMetric(QStyle.PM_SmallIconSize, None, self.gradient_cb)
        self.gradient_cb.setIconSize(QSize(64, icsize))
        model = itemmodels.ContinuousPalettesModel()
        model.setParent(self)
        self.gradient_cb.setModel(model)
        self.gradient_cb.activated[int].connect(self.activated)
        self.gradient_cb.currentIndexChanged.connect(self.currentIndexChanged)
        self.gradient_cb.currentIndexChanged.connect(self.__update_center_visibility)
        form.setWidget(0, QFormLayout.SpanningRole, self.gradient_cb)

        def on_center_spin_value_changed(value):
            if False:
                return 10
            if self.__center != value:
                self.__center = value
                self.centerChanged.emit(self.__center)
        if center is not None:
            self.center_edit = DoubleSpinBox(value=self.__center, minimum=DBL_MIN, maximum=DBL_MAX, minimumStep=0.01, minimumContentsLenght=8, alignment=Qt.AlignRight, stepType=DoubleSpinBox.AdaptiveDecimalStepType, keyboardTracking=False, sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
            self.center_edit.valueChanged.connect(on_center_spin_value_changed)
        else:
            self.center_edit = None
        slider = self.slider = IntervalSlider(int(low * 100), int(high * 100), minimum=0, maximum=100, tickPosition=QSlider.NoTicks, toolTip=self.tr('Low gradient threshold'), whatsThis=self.tr('Applying a low threshold will squeeze the gradient from the lower end'))
        form.addRow(self.tr('Range:'), slider)
        self.slider.intervalChanged.connect(self.__on_slider_moved)
        self.setLayout(form)

    def setModel(self, model: QAbstractItemModel) -> None:
        if False:
            return 10
        self.gradient_cb.setModel(model)

    def model(self) -> QAbstractItemModel:
        if False:
            for i in range(10):
                print('nop')
        return self.gradient_cb.model()

    def findData(self, data: Any, role: Qt.ItemDataRole) -> int:
        if False:
            return 10
        return self.gradient_cb.findData(data, role)

    def setCurrentIndex(self, index: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.gradient_cb.setCurrentIndex(index)
        self.__update_center_visibility()

    def currentIndex(self) -> int:
        if False:
            print('Hello World!')
        return self.gradient_cb.currentIndex()
    currentIndex_ = Property(int, currentIndex, setCurrentIndex, notify=currentIndexChanged)

    def currentData(self, role=Qt.UserRole) -> Any:
        if False:
            print('Hello World!')
        return self.gradient_cb.currentData(role)

    def thresholds(self) -> Tuple[float, float]:
        if False:
            while True:
                i = 10
        return (self.__threshold_low, self.__threshold_high)
    thresholds_ = Property(object, thresholds, notify=thresholdsChanged)

    def thresholdLow(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self.__threshold_low

    def setThresholdLow(self, low: float) -> None:
        if False:
            i = 10
            return i + 15
        self.setThresholds(low, max(self.__threshold_high, low))
    thresholdLow_ = Property(float, thresholdLow, setThresholdLow, notify=thresholdsChanged)

    def thresholdHigh(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self.__threshold_high

    def setThresholdHigh(self, high: float) -> None:
        if False:
            print('Hello World!')
        self.setThresholds(min(self.__threshold_low, high), high)
    thresholdHigh_ = Property(float, thresholdLow, setThresholdLow, notify=thresholdsChanged)

    def __on_slider_moved(self, low: int, high: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        old = (self.__threshold_low, self.__threshold_high)
        self.__threshold_low = low / 100.0
        self.__threshold_high = high / 100.0
        new = (self.__threshold_low, self.__threshold_high)
        if new != old:
            self.thresholdsChanged.emit(*new)

    def setThresholds(self, low: float, high: float) -> None:
        if False:
            i = 10
            return i + 15
        low = round(clip(low, 0.0, 1.0), 2)
        high = round(clip(high, 0.0, 1.0), 2)
        if low > high:
            high = low
        if self.__threshold_low != low or self.__threshold_high != high:
            self.__threshold_high = high
            self.__threshold_low = low
            self.slider.setInterval(int(low * 100), int(high * 100))
            self.thresholdsChanged.emit(high, low)

    def __update_center_visibility(self):
        if False:
            for i in range(10):
                print('nop')
        palette = self.currentData()
        if self.center_edit is None or (visible := (self.center_edit.parent() is not None)) == bool(isinstance(palette, colorpalettes.Palette) and palette.flags & palette.Flags.Diverging):
            return
        if visible:
            self.layout().takeRow(1).labelItem.widget().setParent(None)
            self.center_edit.setParent(None)
        else:
            self.layout().insertRow(1, 'Center at:', self.center_edit)

    def center(self) -> float:
        if False:
            while True:
                i = 10
        return self.__center

    def setCenter(self, center: float) -> None:
        if False:
            i = 10
            return i + 15
        if self.__center != center:
            self.__center = center
            self.center_edit.setValue(center)
            self.centerChanged.emit(center)
    center_ = Property(float, center, setCenter, notify=centerChanged)

def clip(a, amin, amax):
    if False:
        for i in range(10):
            print('nop')
    return min(max(a, amin), amax)