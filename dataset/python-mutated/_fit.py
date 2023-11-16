from typing import Any, Optional, Tuple, Union

class Fit:

    def __init__(self, fit_type: str, fit_args: Tuple[Union[None, float, Any], ...]=()):
        if False:
            return 10
        from ._base import FloatObject, NameObject, NullObject
        self.fit_type = NameObject(fit_type)
        self.fit_args = [NullObject() if a is None or isinstance(a, NullObject) else FloatObject(a) for a in fit_args]

    @classmethod
    def xyz(cls, left: Optional[float]=None, top: Optional[float]=None, zoom: Optional[float]=None) -> 'Fit':
        if False:
            for i in range(10):
                print('nop')
        '\n        Display the page designated by page, with the coordinates (left , top)\n        positioned at the upper-left corner of the window and the contents\n        of the page magnified by the factor zoom.\n\n        A null value for any of the parameters left, top, or zoom specifies\n        that the current value of that parameter is to be retained unchanged.\n\n        A zoom value of 0 has the same meaning as a null value.\n\n        Args:\n            left:\n            top:\n            zoom:\n\n        Returns:\n            The created fit object.\n        '
        return Fit(fit_type='/XYZ', fit_args=(left, top, zoom))

    @classmethod
    def fit(cls) -> 'Fit':
        if False:
            print('Hello World!')
        '\n        Display the page designated by page, with its contents magnified just\n        enough to fit the entire page within the window both horizontally and\n        vertically.\n\n        If the required horizontal and vertical magnification factors are\n        different, use the smaller of the two, centering the page within the\n        window in the other dimension.\n        '
        return Fit(fit_type='/Fit')

    @classmethod
    def fit_horizontally(cls, top: Optional[float]=None) -> 'Fit':
        if False:
            while True:
                i = 10
        '\n        Display the page designated by page , with the vertical coordinate top\n        positioned at the top edge of the window and the contents of the page\n        magnified just enough to fit the entire width of the page within the\n        window.\n\n        A null value for ``top`` specifies that the current value of that\n        parameter is to be retained unchanged.\n\n        Args:\n            top:\n\n        Returns:\n            The created fit object.\n        '
        return Fit(fit_type='/FitH', fit_args=(top,))

    @classmethod
    def fit_vertically(cls, left: Optional[float]=None) -> 'Fit':
        if False:
            print('Hello World!')
        return Fit(fit_type='/FitV', fit_args=(left,))

    @classmethod
    def fit_rectangle(cls, left: Optional[float]=None, bottom: Optional[float]=None, right: Optional[float]=None, top: Optional[float]=None) -> 'Fit':
        if False:
            print('Hello World!')
        '\n        Display the page designated by page , with its contents magnified\n        just enough to fit the rectangle specified by the coordinates\n        left, bottom, right, and top entirely within the window\n        both horizontally and vertically.\n\n        If the required horizontal and vertical magnification factors are\n        different, use the smaller of the two, centering the rectangle within\n        the window in the other dimension.\n\n        A null value for any of the parameters may result in unpredictable\n        behavior.\n\n        Args:\n            left:\n            bottom:\n            right:\n            top:\n\n        Returns:\n            The created fit object.\n        '
        return Fit(fit_type='/FitR', fit_args=(left, bottom, right, top))

    @classmethod
    def fit_box(cls) -> 'Fit':
        if False:
            for i in range(10):
                print('nop')
        '\n        Display the page designated by page , with its contents magnified just\n        enough to fit its bounding box entirely within the window both\n        horizontally and vertically.\n\n        If the required horizontal and vertical magnification factors are\n        different, use the smaller of the two, centering the bounding box\n        within the window in the other dimension.\n        '
        return Fit(fit_type='/FitB')

    @classmethod
    def fit_box_horizontally(cls, top: Optional[float]=None) -> 'Fit':
        if False:
            return 10
        '\n        Display the page designated by page , with the vertical coordinate top\n        positioned at the top edge of the window and the contents of the page\n        magnified just enough to fit the entire width of its bounding box\n        within the window.\n\n        A null value for top specifies that the current value of that parameter\n        is to be retained unchanged.\n\n        Args:\n            top:\n\n        Returns:\n            The created fit object.\n        '
        return Fit(fit_type='/FitBH', fit_args=(top,))

    @classmethod
    def fit_box_vertically(cls, left: Optional[float]=None) -> 'Fit':
        if False:
            while True:
                i = 10
        '\n        Display the page designated by page, with the horizontal coordinate\n        left positioned at the left edge of the window and the contents of the\n        page magnified just enough to fit the entire height of its bounding box\n        within the window.\n\n        A null value for left specifies that the current value of that\n        parameter is to be retained unchanged.\n\n        Args:\n            left:\n\n        Returns:\n            The created fit object.\n        '
        return Fit(fit_type='/FitBV', fit_args=(left,))

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        if not self.fit_args:
            return f'Fit({self.fit_type})'
        return f'Fit({self.fit_type}, {self.fit_args})'
DEFAULT_FIT = Fit.fit()