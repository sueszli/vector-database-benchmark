"""
This object represents the base class for everything related to IO in borb.
It has some convenience methods that allow you to specify how the object
should be persisted, as well as some methods to traverse the object-graph.
"""
import copy
import decimal
import typing
from types import MethodType
import PIL
import borb.io.read.types

class PDFObject:
    """
    This object represents the base class for everything related to IO in borb.
    It has some convenience methods that allow you to specify how the object
    should be persisted, as well as some methods to traverse the object-graph.
    """

    def __init__(self):
        if False:
            return 10
        self._parent: typing.Optional['PDFObject'] = None
        self._is_inline: bool = False
        self._is_unique: bool = False
        self._reference: typing.Optional['Reference'] = None

    @staticmethod
    def _to_json(self, memo_dict={}) -> typing.Any:
        if False:
            print('Hello World!')
        if isinstance(self, bool):
            return self
        if isinstance(self, borb.io.read.types.Boolean):
            return bool(self)
        if isinstance(self, borb.io.read.types.CanvasOperatorName):
            return str(self)
        if isinstance(self, borb.io.read.types.Decimal):
            return float(self)
        if isinstance(self, decimal.Decimal):
            return float(self)
        if isinstance(self, float) or isinstance(self, int):
            return self
        if isinstance(self, bytes):
            return str(self)
        if isinstance(self, borb.io.read.types.Dictionary):
            out: typing.Dict[str, typing.Any] = {}
            memo_dict[id(self)] = out
            for (k, v) in self.items():
                out[str(k)] = PDFObject._to_json(v, memo_dict)
            return out
        if isinstance(self, dict):
            dict_out: typing.Dict[str, typing.Any] = {}
            memo_dict[id(self)] = dict_out
            for (k, v) in self.items():
                dict_out[str(k)] = PDFObject._to_json(v, memo_dict)
            return dict_out
        if isinstance(self, borb.io.read.types.Element):
            from borb.io.read.types import ET
            return str(ET.tostring(self))
        if isinstance(self, borb.io.read.types.Name):
            return str(self)
        if isinstance(self, borb.io.read.types.String):
            return str(self)
        if isinstance(self, borb.io.read.types.List):
            list_out: typing.List[typing.Any] = []
            memo_dict[id(self)] = list_out
            for v in self:
                list_out.append(PDFObject._to_json(v, memo_dict))
            return list_out
        if isinstance(self, borb.io.read.types.Reference):
            return '%d %d R' % (self.generation_number or 0, self.object_number or 0)
        return None

    @staticmethod
    def add_pdf_object_methods(non_borb_object: typing.Any) -> typing.Any:
        if False:
            i = 10
            return i + 15
        '\n        This method allows you to pretend an object is actually a PDFObject.\n        It adds all the methods that are present for a PDFObject.\n        It also adds a utility hashing method for images (since PIL normally does not hash images)\n        :param non_borb_object:\n        :return:\n        '

        def _deepcopy_and_add_methods(self, memodict={}):
            if False:
                i = 10
                return i + 15
            prev_function_ptr = self.__deepcopy__
            self.__deepcopy__ = None
            out = copy.deepcopy(self, memodict)
            self.__deepcopy__ = prev_function_ptr
            PDFObject.add_pdf_object_methods(out)
            return out

        def _get_parent(self) -> typing.Optional[PDFObject]:
            if False:
                return 10
            if '_parent' not in vars(self):
                setattr(self, '_parent', None)
            return self._parent

        def _get_reference(self) -> typing.Optional['Reference']:
            if False:
                while True:
                    i = 10
            if '_reference' not in vars(self):
                setattr(self, '_reference', None)
            return self._reference

        def _get_root(self) -> PDFObject:
            if False:
                return 10
            p = self
            while p.get_parent() is not None:
                p = p.get_parent()
            return p

        def _is_inline(self) -> bool:
            if False:
                while True:
                    i = 10
            if '_is_inline' not in vars(self):
                setattr(self, '_is_inline', False)
            return self._is_inline

        def _is_unique(self) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            if '_is_unique' not in vars(self):
                setattr(self, '_is_unique', False)
            return self._is_unique

        def _pil_image_hash(self):
            if False:
                return 10
            w = self.width
            h = self.height
            pixels = [self.getpixel((0, 0)), self.getpixel((0, h - 1)), self.getpixel((w - 1, 0)), self.getpixel((w - 1, h - 1))]
            hashcode = 1
            for p in pixels:
                if isinstance(p, typing.List) or isinstance(p, typing.Tuple):
                    hashcode += 32 * hashcode + sum(p)
                else:
                    hashcode += 32 * hashcode + p
            return hashcode

        def _set_is_inline(self, is_inline: bool) -> PDFObject:
            if False:
                print('Hello World!')
            if '_is_inline' not in vars(self):
                setattr(self, '_is_inline', False)
            self._is_inline = is_inline
            return self

        def _set_is_unique(self, is_unique: bool) -> PDFObject:
            if False:
                while True:
                    i = 10
            if '_is_unique' not in vars(self):
                setattr(self, '_is_unique', False)
            self._is_unique = is_unique
            return self

        def _set_parent(self, parent: PDFObject) -> PDFObject:
            if False:
                print('Hello World!')
            if '_parent' not in vars(self):
                setattr(self, '_parent', None)
            self._parent = parent
            return self

        def _set_reference(self, reference: 'Reference') -> PDFObject:
            if False:
                print('Hello World!')
            if '_reference' not in vars(self):
                setattr(self, '_reference', None)
            self._reference = reference
            return self
        non_borb_object.set_parent = MethodType(_set_parent, non_borb_object)
        non_borb_object.get_parent = MethodType(_get_parent, non_borb_object)
        non_borb_object.get_root = MethodType(_get_root, non_borb_object)
        non_borb_object.set_reference = MethodType(_set_reference, non_borb_object)
        non_borb_object.get_reference = MethodType(_get_reference, non_borb_object)
        non_borb_object.set_is_inline = MethodType(_set_is_inline, non_borb_object)
        non_borb_object.is_inline = MethodType(_is_inline, non_borb_object)
        non_borb_object.set_is_unique = MethodType(_set_is_unique, non_borb_object)
        non_borb_object.is_unique = MethodType(_is_unique, non_borb_object)
        non_borb_object.__deepcopy__ = MethodType(_deepcopy_and_add_methods, non_borb_object)
        if isinstance(non_borb_object, PIL.Image.Image):
            non_borb_object.__hash__ = MethodType(_pil_image_hash, non_borb_object)
        return non_borb_object

    def get_parent(self) -> typing.Optional['PDFObject']:
        if False:
            print('Hello World!')
        '\n        This function returns the parent of this PDFObject, or None if no such PDFObject exists\n        :return:    the parent of this PDFObject\n        '
        return self._parent

    def get_reference(self) -> typing.Optional['Reference']:
        if False:
            while True:
                i = 10
        '\n        This function gets the Reference being used for this PDFObject\n        :return:    the Reference being used for this PDFObject\n        '
        return self._reference

    def get_root(self) -> typing.Optional['PDFObject']:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns the root (of the parent hierarchy) of this PDFObject, or None if no such PDFObject exists\n        :return:    the root of this PDFObject\n        '
        p: typing.Optional['PDFObject'] = self
        while p is not None and p.get_parent() is not None:
            p = p.get_parent()
        return p

    def is_inline(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns True if this PDFObject should be persisted inline, False otherwise\n        :return:    whether this PDFObject should be persisted online\n        '
        return self._is_inline

    def is_unique(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        This function returns True if this PDFObject should always be treated\n        as if it is unique (for IO purposes), regardless of hashing equality.\n        :return:    whether this PDFObject is unique\n        '
        return self._is_unique

    def set_is_inline(self, is_inline: bool) -> 'PDFObject':
        if False:
            while True:
                i = 10
        '\n        This function sets the is_inline flag of this PDFObject.\n        An inline object is always persisted immediately when needed, it is never turned into a reference.\n        :param is_inline:   whether this PDFObject should be persisted inline, or not\n        :return:            self\n        '
        self._is_inline = is_inline
        return self

    def set_is_unique(self, is_unique: bool) -> 'PDFObject':
        if False:
            print('Hello World!')
        '\n        This function sets the is_unique flag of this PDFObject.\n        A unique object is always persisted as itself,\n        or its own reference even if it should be equal to another PDFObject.\n        :param is_unique:   whether this PDFObject should be unique or not\n        :return:            self\n        '
        self._is_unique = is_unique
        return self

    def set_parent(self, parent: 'PDFObject') -> 'PDFObject':
        if False:
            while True:
                i = 10
        '\n        This function sets the parent (PDFObject) of this PDFObject\n        :param parent:  the parent (PDFObject)\n        :return:        self\n        '
        self._parent = parent
        return self

    def set_reference(self, reference: 'Reference') -> 'PDFObject':
        if False:
            print('Hello World!')
        '\n        This function sets the Reference to be used for this PDFObject\n        :param reference:   the Reference to be used for this PDFObject\n        :return:            self\n        '
        self._reference = reference
        return self

    def to_json(self) -> typing.Any:
        if False:
            i = 10
            return i + 15
        '\n        This function converts this PDFObject into a set of nested dictionaries, lists and primitives\n        :return:    a JSON-like object\n        '
        return PDFObject._to_json(self)