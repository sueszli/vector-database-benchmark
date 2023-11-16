from typing import Any, List, Optional
from ._base import BooleanObject, NameObject, NumberObject
from ._data_structures import ArrayObject, DictionaryObject
f_obj = BooleanObject(False)

class ViewerPreferences(DictionaryObject):

    def _get_bool(self, key: str, deft: Optional[BooleanObject]) -> BooleanObject:
        if False:
            while True:
                i = 10
        return self.get(key, deft)

    def _set_bool(self, key: str, v: bool) -> None:
        if False:
            i = 10
            return i + 15
        self[NameObject(key)] = BooleanObject(v is True)

    def _get_name(self, key: str, deft: Optional[NameObject]) -> Optional[NameObject]:
        if False:
            print('Hello World!')
        return self.get(key, deft)

    def _set_name(self, key: str, lst: List[str], v: NameObject) -> None:
        if False:
            i = 10
            return i + 15
        if v[0] != '/':
            raise ValueError(f"{v} is not starting with '/'")
        if lst != [] and v not in lst:
            raise ValueError(f'{v} is not par of acceptable values')
        self[NameObject(key)] = NameObject(v)

    def _get_arr(self, key: str, deft: Optional[List[Any]]) -> NumberObject:
        if False:
            i = 10
            return i + 15
        return self.get(key, None if deft is None else ArrayObject(deft))

    def _set_arr(self, key: str, v: Optional[ArrayObject]) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(v, ArrayObject):
            raise ValueError('ArrayObject is expected')
        self[NameObject(key)] = v

    def _get_int(self, key: str, deft: Optional[NumberObject]) -> NumberObject:
        if False:
            for i in range(10):
                print('nop')
        return self.get(key, deft)

    def _set_int(self, key: str, v: int) -> None:
        if False:
            print('Hello World!')
        self[NameObject(key)] = NumberObject(v)

    def __new__(cls: Any, value: Any=None) -> 'ViewerPreferences':
        if False:
            for i in range(10):
                print('nop')

        def _add_prop_bool(key: str, deft: Optional[BooleanObject]) -> property:
            if False:
                print('Hello World!')
            return property(lambda self: self._get_bool(key, deft), lambda self, v: self._set_bool(key, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined\n            ')

        def _add_prop_name(key: str, lst: List[str], deft: Optional[NameObject]) -> property:
            if False:
                while True:
                    i = 10
            return property(lambda self: self._get_name(key, deft), lambda self, v: self._set_name(key, lst, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined.\n            Acceptable values: {lst}\n            ')

        def _add_prop_arr(key: str, deft: Optional[ArrayObject]) -> property:
            if False:
                for i in range(10):
                    print('nop')
            return property(lambda self: self._get_arr(key, deft), lambda self, v: self._set_arr(key, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined\n            ')

        def _add_prop_int(key: str, deft: Optional[int]) -> property:
            if False:
                i = 10
                return i + 15
            return property(lambda self: self._get_int(key, deft), lambda self, v: self._set_int(key, v), None, f'\n            Returns/Modify the status of {key}, Returns {deft} if not defined\n            ')
        cls.hide_toolbar = _add_prop_bool('/HideToolbar', f_obj)
        cls.hide_menubar = _add_prop_bool('/HideMenubar', f_obj)
        cls.hide_windowui = _add_prop_bool('/HideWindowUI', f_obj)
        cls.fit_window = _add_prop_bool('/FitWindow', f_obj)
        cls.center_window = _add_prop_bool('/CenterWindow', f_obj)
        cls.display_doctitle = _add_prop_bool('/DisplayDocTitle', f_obj)
        cls.non_fullscreen_pagemode = _add_prop_name('/NonFullScreenPageMode', ['/UseNone', '/UseOutlines', '/UseThumbs', '/UseOC'], NameObject('/UseNone'))
        cls.direction = _add_prop_name('/Direction', ['/L2R', '/R2L'], NameObject('/L2R'))
        cls.view_area = _add_prop_name('/ViewArea', [], None)
        cls.view_clip = _add_prop_name('/ViewClip', [], None)
        cls.print_area = _add_prop_name('/PrintArea', [], None)
        cls.print_clip = _add_prop_name('/PrintClip', [], None)
        cls.print_scaling = _add_prop_name('/PrintScaling', [], None)
        cls.duplex = _add_prop_name('/Duplex', ['/Simplex', '/DuplexFlipShortEdge', '/DuplexFlipLongEdge'], None)
        cls.pick_tray_by_pdfsize = _add_prop_bool('/PickTrayByPDFSize', None)
        cls.print_pagerange = _add_prop_arr('/PrintPageRange', None)
        cls.num_copies = _add_prop_int('/NumCopies', None)
        return DictionaryObject.__new__(cls)

    def __init__(self, obj: Optional[DictionaryObject]=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(self)
        if obj is not None:
            self.update(obj.items())
        try:
            self.indirect_reference = obj.indirect_reference
        except AttributeError:
            pass