import math
from decimal import Decimal
import numpy as np
from AnyQt.QtCore import QLocale, QSize
from AnyQt.QtWidgets import QDoubleSpinBox, QStyle, QStyleOptionSpinBox
DBL_MIN = float(np.finfo(float).min)
DBL_MAX = float(np.finfo(float).max)
DBL_MAX_10_EXP = math.floor(math.log10(DBL_MAX))
DBL_DIG = math.floor(math.log10(2 ** np.finfo(float).nmant))

class DoubleSpinBox(QDoubleSpinBox):
    """
    A QDoubleSpinSubclass with non-fixed decimal precision/rounding.
    """

    def __init__(self, parent=None, decimals=-1, minimumStep=1e-05, minimumContentsLenght=-1, **kwargs):
        if False:
            while True:
                i = 10
        self.__decimals = decimals
        self.__minimumStep = minimumStep
        self.__minimumContentsLength = minimumContentsLenght
        stepType = kwargs.pop('stepType', DoubleSpinBox.DefaultStepType)
        super().__init__(parent, **kwargs)
        if decimals < 0:
            super().setDecimals(DBL_MAX_10_EXP + DBL_DIG)
        else:
            super().setDecimals(decimals)
        self.setStepType(stepType)

    def setDecimals(self, prec: int) -> None:
        if False:
            print('Hello World!')
        '\n        Set the number of decimals in display/edit\n\n        If negative value then no rounding takes place and the value is\n        displayed using `QLocale.FloatingPointShortest` precision.\n        '
        self.__decimals = prec
        if prec < 0:
            super().setDecimals(DBL_MAX_10_EXP + DBL_DIG)
        else:
            super().setDecimals(prec)

    def decimals(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__decimals

    def setMinimumStep(self, step):
        if False:
            return 10
        '\n        Minimum step size when `stepType() == AdaptiveDecimalStepType`\n        and `decimals() < 0`.\n        '
        self.__minimumStep = step

    def minimumStep(self):
        if False:
            print('Hello World!')
        return self.__minimumStep

    def textFromValue(self, v: float) -> str:
        if False:
            return 10
        'Reimplemented.'
        if self.__decimals < 0:
            locale = self.locale()
            return locale.toString(v, 'f', QLocale.FloatingPointShortest)
        else:
            return super().textFromValue(v)

    def stepBy(self, steps: int) -> None:
        if False:
            return 10
        '\n        Reimplemented.\n        '
        value = self.value()
        value_dec = Decimal(str(value))
        if self.stepType() == DoubleSpinBox.AdaptiveDecimalStepType:
            step_dec = self.__adaptiveDecimalStep(steps)
        else:
            step_dec = Decimal(str(self.singleStep()))
        value_dec = value_dec + step_dec * steps
        self.setValue(float(value_dec))

    def __adaptiveDecimalStep(self, steps: int) -> Decimal:
        if False:
            for i in range(10):
                print('nop')
        decValue: Decimal = Decimal(str(self.value()))
        decimals = self.__decimals
        if decimals < 0:
            minStep = Decimal(str(self.__minimumStep))
        else:
            minStep = Decimal(10) ** (-decimals)
        absValue = abs(decValue)
        if absValue < minStep:
            return minStep
        valueNegative = decValue < 0
        stepsNegative = steps < 0
        if valueNegative != stepsNegative:
            absValue /= Decimal('1.01')
        step = Decimal(10) ** (math.floor(absValue.log10()) - 1)
        return max(minStep, step)
    if not hasattr(QDoubleSpinBox, 'stepType'):
        DefaultStepType = 0
        AdaptiveDecimalStepType = 1
        __stepType = AdaptiveDecimalStepType

        def setStepType(self, stepType):
            if False:
                return 10
            self.__stepType = stepType

        def stepType(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.__stepType

    def setMinimumContentsLength(self, characters: int):
        if False:
            i = 10
            return i + 15
        self.__minimumContentsLength = characters
        self.updateGeometry()

    def minimumContentsLength(self):
        if False:
            while True:
                i = 10
        return self.__minimumContentsLength

    def sizeHint(self) -> QSize:
        if False:
            for i in range(10):
                print('nop')
        if self.minimumContentsLength() < 0:
            return super().sizeHint()
        self.ensurePolished()
        fm = self.fontMetrics()
        template = 'X' * self.minimumContentsLength()
        template += '.'
        if self.prefix():
            template = self.prefix() + ' ' + template
        if self.suffix():
            template = template + self.suffix()
        if self.minimum() < 0.0:
            template = '-' + template
        if self.specialValueText():
            templates = [template, self.specialValueText()]
        else:
            templates = [template]
        height = self.lineEdit().sizeHint().height()
        width = max(map(fm.horizontalAdvance, templates))
        width += 2
        hint = QSize(width, height)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)
        sh = self.style().sizeFromContents(QStyle.CT_SpinBox, opt, hint, self)
        return sh

    def minimumSizeHint(self) -> QSize:
        if False:
            return 10
        if self.minimumContentsLength() < 0:
            return super().minimumSizeHint()
        else:
            return self.sizeHint()