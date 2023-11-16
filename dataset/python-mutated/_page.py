import math
import re
import warnings
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union, cast, overload
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfReaderProtocol, PdfWriterProtocol
from ._text_extraction import OrientationNotFoundError, crlf_space_check, handle_tj, mult
from ._utils import WHITESPACES_AS_REGEXP, CompressedTransformationMatrix, File, ImageFile, TransformationMatrixType, deprecation_no_replacement, deprecation_with_replacement, logger_warning, matrix_multiply
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Ressources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import ArrayObject, ContentStream, DictionaryObject, EncodedStreamObject, FloatObject, IndirectObject, NameObject, NullObject, NumberObject, RectangleObject, StreamObject
MERGE_CROP_BOX = 'cropbox'

def _get_rectangle(self: Any, name: str, defaults: Iterable[str]) -> RectangleObject:
    if False:
        while True:
            i = 10
    retval: Union[None, RectangleObject, IndirectObject] = self.get(name)
    if isinstance(retval, RectangleObject):
        return retval
    if retval is None:
        for d in defaults:
            retval = self.get(d)
            if retval is not None:
                break
    if isinstance(retval, IndirectObject):
        retval = self.pdf.get_object(retval)
    retval = RectangleObject(retval)
    _set_rectangle(self, name, retval)
    return retval

def getRectangle(self: Any, name: str, defaults: Iterable[str]) -> RectangleObject:
    if False:
        while True:
            i = 10
    deprecation_no_replacement('getRectangle', '3.0.0')
    return _get_rectangle(self, name, defaults)

def _set_rectangle(self: Any, name: str, value: Union[RectangleObject, float]) -> None:
    if False:
        for i in range(10):
            print('nop')
    name = NameObject(name)
    self[name] = value

def setRectangle(self: Any, name: str, value: Union[RectangleObject, float]) -> None:
    if False:
        print('Hello World!')
    deprecation_no_replacement('setRectangle', '3.0.0')
    _set_rectangle(self, name, value)

def _delete_rectangle(self: Any, name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    del self[name]

def deleteRectangle(self: Any, name: str) -> None:
    if False:
        i = 10
        return i + 15
    deprecation_no_replacement('deleteRectangle', '3.0.0')
    del self[name]

def _create_rectangle_accessor(name: str, fallback: Iterable[str]) -> property:
    if False:
        for i in range(10):
            print('nop')
    return property(lambda self: _get_rectangle(self, name, fallback), lambda self, value: _set_rectangle(self, name, value), lambda self: _delete_rectangle(self, name))

def createRectangleAccessor(name: str, fallback: Iterable[str]) -> property:
    if False:
        for i in range(10):
            print('nop')
    deprecation_no_replacement('createRectangleAccessor', '3.0.0')
    return _create_rectangle_accessor(name, fallback)

class Transformation:
    """
    Represent a 2D transformation.

    The transformation between two coordinate systems is represented by a 3-by-3
    transformation matrix matrix with the following form::

        a b 0
        c d 0
        e f 1

    Because a transformation matrix has only six elements that can be changed,
    it is usually specified in PDF as the six-element array [ a b c d e f ].

    Coordinate transformations are expressed as matrix multiplications::

                                 a b 0
     [ x′ y′ 1 ] = [ x y 1 ] ×   c d 0
                                 e f 1


    Example:
        >>> from pypdf import Transformation
        >>> op = Transformation().scale(sx=2, sy=3).translate(tx=10, ty=20)
        >>> page.add_transformation(op)
    """

    def __init__(self, ctm: CompressedTransformationMatrix=(1, 0, 0, 1, 0, 0)):
        if False:
            print('Hello World!')
        self.ctm = ctm

    @property
    def matrix(self) -> TransformationMatrixType:
        if False:
            return 10
        '\n        Return the transformation matrix as a tuple of tuples in the form:\n\n        ((a, b, 0), (c, d, 0), (e, f, 1))\n        '
        return ((self.ctm[0], self.ctm[1], 0), (self.ctm[2], self.ctm[3], 0), (self.ctm[4], self.ctm[5], 1))

    @staticmethod
    def compress(matrix: TransformationMatrixType) -> CompressedTransformationMatrix:
        if False:
            i = 10
            return i + 15
        '\n        Compresses the transformation matrix into a tuple of (a, b, c, d, e, f).\n\n        Args:\n            matrix: The transformation matrix as a tuple of tuples.\n\n        Returns:\n            A tuple representing the transformation matrix as (a, b, c, d, e, f)\n        '
        return (matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1], matrix[2][0], matrix[2][1])

    def transform(self, m: 'Transformation') -> 'Transformation':
        if False:
            i = 10
            return i + 15
        '\n        Apply one transformation to another.\n\n        Args:\n            m: a Transformation to apply.\n\n        Returns:\n            A new ``Transformation`` instance\n\n        Example:\n            >>> from pypdf import Transformation\n            >>> op = Transformation((1, 0, 0, -1, 0, height)) # vertical mirror\n            >>> op = Transformation().transform(Transformation((-1, 0, 0, 1, iwidth, 0))) # horizontal mirror\n            >>> page.add_transformation(op)\n        '
        ctm = Transformation.compress(matrix_multiply(self.matrix, m.matrix))
        return Transformation(ctm)

    def translate(self, tx: float=0, ty: float=0) -> 'Transformation':
        if False:
            while True:
                i = 10
        '\n        Translate the contents of a page.\n\n        Args:\n            tx: The translation along the x-axis.\n            ty: The translation along the y-axis.\n\n        Returns:\n            A new ``Transformation`` instance\n        '
        m = self.ctm
        return Transformation(ctm=(m[0], m[1], m[2], m[3], m[4] + tx, m[5] + ty))

    def scale(self, sx: Optional[float]=None, sy: Optional[float]=None) -> 'Transformation':
        if False:
            while True:
                i = 10
        '\n        Scale the contents of a page towards the origin of the coordinate system.\n\n        Typically, that is the lower-left corner of the page. That can be\n        changed by translating the contents / the page boxes.\n\n        Args:\n            sx: The scale factor along the x-axis.\n            sy: The scale factor along the y-axis.\n\n        Returns:\n            A new Transformation instance with the scaled matrix.\n        '
        if sx is None and sy is None:
            raise ValueError('Either sx or sy must be specified')
        if sx is None:
            sx = sy
        if sy is None:
            sy = sx
        assert sx is not None
        assert sy is not None
        op: TransformationMatrixType = ((sx, 0, 0), (0, sy, 0), (0, 0, 1))
        ctm = Transformation.compress(matrix_multiply(self.matrix, op))
        return Transformation(ctm)

    def rotate(self, rotation: float) -> 'Transformation':
        if False:
            i = 10
            return i + 15
        '\n        Rotate the contents of a page.\n\n        Args:\n            rotation: The angle of rotation in degrees.\n\n        Returns:\n            A new ``Transformation`` instance with the rotated matrix.\n        '
        rotation = math.radians(rotation)
        op: TransformationMatrixType = ((math.cos(rotation), math.sin(rotation), 0), (-math.sin(rotation), math.cos(rotation), 0), (0, 0, 1))
        ctm = Transformation.compress(matrix_multiply(self.matrix, op))
        return Transformation(ctm)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Transformation(ctm={self.ctm})'

    @overload
    def apply_on(self, pt: List[float], as_object: bool=False) -> List[float]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def apply_on(self, pt: Tuple[float, float], as_object: bool=False) -> Tuple[float, float]:
        if False:
            return 10
        ...

    def apply_on(self, pt: Union[Tuple[float, float], List[float]], as_object: bool=False) -> Union[Tuple[float, float], List[float]]:
        if False:
            return 10
        "\n        Apply the transformation matrix on the given point.\n\n        Args:\n            pt: A tuple or list representing the point in the form (x, y)\n\n        Returns:\n            A tuple or list representing the transformed point in the form (x', y')\n        "
        typ = FloatObject if as_object else float
        pt1 = (typ(float(pt[0]) * self.ctm[0] + float(pt[1]) * self.ctm[2] + self.ctm[4]), typ(float(pt[0]) * self.ctm[1] + float(pt[1]) * self.ctm[3] + self.ctm[5]))
        return list(pt1) if isinstance(pt, list) else pt1

class PageObject(DictionaryObject):
    """
    PageObject represents a single page within a PDF file.

    Typically this object will be created by accessing the
    :meth:`get_page()<pypdf.PdfReader.get_page>` method of the
    :class:`PdfReader<pypdf.PdfReader>` class, but it is
    also possible to create an empty page with the
    :meth:`create_blank_page()<pypdf._page.PageObject.create_blank_page>` static method.

    Args:
        pdf: PDF file the page belongs to.
        indirect_reference: Stores the original indirect reference to
            this object in its source PDF
    """
    original_page: 'PageObject'

    def __init__(self, pdf: Union[None, PdfReaderProtocol, PdfWriterProtocol]=None, indirect_reference: Optional[IndirectObject]=None, indirect_ref: Optional[IndirectObject]=None) -> None:
        if False:
            i = 10
            return i + 15
        DictionaryObject.__init__(self)
        self.pdf: Union[None, PdfReaderProtocol, PdfWriterProtocol] = pdf
        self.inline_images: Optional[Dict[str, ImageFile]] = None
        self.inline_images_keys: Optional[List[Union[str, List[str]]]] = None
        if indirect_ref is not None:
            warnings.warn('indirect_ref is deprecated and will be removed in pypdf 4.0.0. Use indirect_reference instead of indirect_ref.', DeprecationWarning)
            if indirect_reference is not None:
                raise ValueError('Use indirect_reference instead of indirect_ref.')
            indirect_reference = indirect_ref
        self.indirect_reference = indirect_reference

    @property
    def indirect_ref(self) -> Optional[IndirectObject]:
        if False:
            i = 10
            return i + 15
        warnings.warn('indirect_ref is deprecated and will be removed in pypdf 4.0.0Use indirect_reference instead of indirect_ref.', DeprecationWarning)
        return self.indirect_reference

    @indirect_ref.setter
    def indirect_ref(self, value: Optional[IndirectObject]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.indirect_reference = value

    def hash_value_data(self) -> bytes:
        if False:
            print('Hello World!')
        data = super().hash_value_data()
        data += b'%d' % id(self)
        return data

    @property
    def user_unit(self) -> float:
        if False:
            return 10
        '\n        A read-only positive number giving the size of user space units.\n\n        It is in multiples of 1/72 inch. Hence a value of 1 means a user\n        space unit is 1/72 inch, and a value of 3 means that a user\n        space unit is 3/72 inch.\n        '
        return self.get(PG.USER_UNIT, 1)

    @staticmethod
    def create_blank_page(pdf: Union[None, PdfReaderProtocol, PdfWriterProtocol]=None, width: Union[float, Decimal, None]=None, height: Union[float, Decimal, None]=None) -> 'PageObject':
        if False:
            print('Hello World!')
        '\n        Return a new blank page.\n\n        If ``width`` or ``height`` is ``None``, try to get the page size\n        from the last page of *pdf*.\n\n        Args:\n            pdf: PDF file the page belongs to\n            width: The width of the new page expressed in default user\n                space units.\n            height: The height of the new page expressed in default user\n                space units.\n\n        Returns:\n            The new blank page\n\n        Raises:\n            PageSizeNotDefinedError: if ``pdf`` is ``None`` or contains\n                no page\n        '
        page = PageObject(pdf)
        page.__setitem__(NameObject(PG.TYPE), NameObject('/Page'))
        page.__setitem__(NameObject(PG.PARENT), NullObject())
        page.__setitem__(NameObject(PG.RESOURCES), DictionaryObject())
        if width is None or height is None:
            if pdf is not None and len(pdf.pages) > 0:
                lastpage = pdf.pages[len(pdf.pages) - 1]
                width = lastpage.mediabox.width
                height = lastpage.mediabox.height
            else:
                raise PageSizeNotDefinedError
        page.__setitem__(NameObject(PG.MEDIABOX), RectangleObject((0, 0, width, height)))
        return page

    @staticmethod
    def createBlankPage(pdf: Optional[PdfReaderProtocol]=None, width: Union[float, Decimal, None]=None, height: Union[float, Decimal, None]=None) -> 'PageObject':
        if False:
            return 10
        '\n        Use :meth:`create_blank_page` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('createBlankPage', 'create_blank_page', '3.0.0')
        return PageObject.create_blank_page(pdf, width, height)

    @property
    def _old_images(self) -> List[File]:
        if False:
            while True:
                i = 10
        "\n        Get a list of all images of the page.\n\n        This requires pillow. You can install it via 'pip install pypdf[image]'.\n\n        For the moment, this does NOT include inline images. They will be added\n        in future.\n        "
        images_extracted: List[File] = []
        if RES.XOBJECT not in self[PG.RESOURCES]:
            return images_extracted
        x_object = self[PG.RESOURCES][RES.XOBJECT].get_object()
        for obj in x_object:
            if x_object[obj][IA.SUBTYPE] == '/Image':
                (extension, byte_stream, img) = _xobj_to_image(x_object[obj])
                if extension is not None:
                    filename = f'{obj[1:]}{extension}'
                    images_extracted.append(File(name=filename, data=byte_stream))
                    images_extracted[-1].image = img
                    images_extracted[-1].indirect_reference = x_object[obj].indirect_reference
        return images_extracted

    def _get_ids_image(self, obj: Optional[DictionaryObject]=None, ancest: Optional[List[str]]=None, call_stack: Optional[List[Any]]=None) -> List[Union[str, List[str]]]:
        if False:
            for i in range(10):
                print('nop')
        if call_stack is None:
            call_stack = []
        _i = getattr(obj, 'indirect_reference', None)
        if _i in call_stack:
            return []
        else:
            call_stack.append(_i)
        if self.inline_images_keys is None:
            nb_inlines = len(re.findall(WHITESPACES_AS_REGEXP + b'BI' + WHITESPACES_AS_REGEXP, self._get_contents_as_bytes() or b''))
            self.inline_images_keys = [f'~{x}~' for x in range(nb_inlines)]
        if obj is None:
            obj = self
        if ancest is None:
            ancest = []
        lst: List[Union[str, List[str]]] = []
        if PG.RESOURCES not in obj or RES.XOBJECT not in cast(DictionaryObject, obj[PG.RESOURCES]):
            return self.inline_images_keys
        x_object = obj[PG.RESOURCES][RES.XOBJECT].get_object()
        for o in x_object:
            if not isinstance(x_object[o], StreamObject):
                continue
            if x_object[o][IA.SUBTYPE] == '/Image':
                lst.append(o if len(ancest) == 0 else ancest + [o])
            else:
                lst.extend(self._get_ids_image(x_object[o], ancest + [o], call_stack))
        return lst + self.inline_images_keys

    def _get_image(self, id: Union[str, List[str], Tuple[str]], obj: Optional[DictionaryObject]=None) -> ImageFile:
        if False:
            print('Hello World!')
        if obj is None:
            obj = cast(DictionaryObject, self)
        if isinstance(id, tuple):
            id = list(id)
        if isinstance(id, List) and len(id) == 1:
            id = id[0]
        try:
            xobjs = cast(DictionaryObject, cast(DictionaryObject, obj[PG.RESOURCES])[RES.XOBJECT])
        except KeyError:
            if not (id[0] == '~' and id[-1] == '~'):
                raise
        if isinstance(id, str):
            if id[0] == '~' and id[-1] == '~':
                if self.inline_images is None:
                    self.inline_images = self._get_inline_images()
                if self.inline_images is None:
                    raise KeyError('no inline image can be found')
                return self.inline_images[id]
            imgd = _xobj_to_image(cast(DictionaryObject, xobjs[id]))
            (extension, byte_stream) = imgd[:2]
            f = ImageFile(name=f'{id[1:]}{extension}', data=byte_stream, image=imgd[2], indirect_reference=xobjs[id].indirect_reference)
            return f
        else:
            ids = id[1:]
            return self._get_image(ids, cast(DictionaryObject, xobjs[id[0]]))

    @property
    def images(self) -> List[ImageFile]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Read-only property emulating a list of images on a page.\n\n        Get a list of all images on the page. The key can be:\n        - A string (for the top object)\n        - A tuple (for images within XObject forms)\n        - An integer\n\n        Examples:\n            reader.pages[0].images[0]        # return fist image\n            reader.pages[0].images[\'/I0\']    # return image \'/I0\'\n            # return image \'/Image1\' within \'/TP1\' Xobject/Form:\n            reader.pages[0].images[\'/TP1\',\'/Image1\']\n            for img in reader.pages[0].images: # loop within all objects\n\n        images.keys() and images.items() can be used.\n\n        The ImageFile has the following properties:\n\n            `.name` : name of the object\n            `.data` : bytes of the object\n            `.image`  : PIL Image Object\n            `.indirect_reference` : object reference\n\n        and the following methods:\n            `.replace(new_image: PIL.Image.Image, **kwargs)` :\n                replace the image in the pdf with the new image\n                applying the saving parameters indicated (such as quality)\n\n        Example usage:\n\n            reader.pages[0].images[0]=replace(Image.open("new_image.jpg", quality = 20)\n\n        Inline images are extracted and named ~0~, ~1~, ..., with the\n        indirect_reference set to None.\n        '
        return _VirtualListImages(self._get_ids_image, self._get_image)

    def _get_inline_images(self) -> Dict[str, ImageFile]:
        if False:
            print('Hello World!')
        '\n        get inline_images\n        entries will be identified as ~1~\n        '
        content = self.get_contents()
        if content is None:
            return {}
        imgs_data = []
        for (param, ope) in content.operations:
            if ope == b'INLINE IMAGE':
                imgs_data.append({'settings': param['settings'], '__streamdata__': param['data']})
            elif ope in (b'BI', b'EI', b'ID'):
                raise PdfReadError(f'{ope} operator met whereas not expected,please share usecase with pypdf dev team')
            'backup\n            elif ope == b"BI":\n                img_data["settings"] = {}\n            elif ope == b"EI":\n                imgs_data.append(img_data)\n                img_data = {}\n            elif ope == b"ID":\n                img_data["__streamdata__"] = b""\n            elif "__streamdata__" in img_data:\n                if len(img_data["__streamdata__"]) > 0:\n                    img_data["__streamdata__"] += b"\n"\n                    raise Exception("check append")\n                img_data["__streamdata__"] += param\n            elif "settings" in img_data:\n                img_data["settings"][ope.decode()] = param\n            '
        files = {}
        for (num, ii) in enumerate(imgs_data):
            init = {'__streamdata__': ii['__streamdata__'], '/Length': len(ii['__streamdata__'])}
            for (k, v) in ii['settings'].items():
                try:
                    v = NameObject({'/G': '/DeviceGray', '/RGB': '/DeviceRGB', '/CMYK': '/DeviceCMYK', '/I': '/Indexed', '/AHx': '/ASCIIHexDecode', '/A85': '/ASCII85Decode', '/LZW': '/LZWDecode', '/Fl': '/FlateDecode', '/RL': '/RunLengthDecode', '/CCF': '/CCITTFaxDecode', '/DCT': '/DCTDecode'}[v])
                except (TypeError, KeyError):
                    if isinstance(v, NameObject):
                        try:
                            res = cast(DictionaryObject, self['/Resources'])['/ColorSpace']
                            v = cast(DictionaryObject, res)[v]
                        except KeyError:
                            raise PdfReadError(f'Can not find resource entry {v} for {k}')
                init[NameObject({'/BPC': '/BitsPerComponent', '/CS': '/ColorSpace', '/D': '/Decode', '/DP': '/DecodeParms', '/F': '/Filter', '/H': '/Height', '/W': '/Width', '/I': '/Interpolate', '/Intent': '/Intent', '/IM': '/ImageMask'}[k])] = v
            ii['object'] = EncodedStreamObject.initialize_from_dictionary(init)
            (extension, byte_stream, img) = _xobj_to_image(ii['object'])
            files[f'~{num}~'] = ImageFile(name=f'~{num}~{extension}', data=byte_stream, image=img, indirect_reference=None)
        return files

    @property
    def rotation(self) -> int:
        if False:
            return 10
        '\n        The VISUAL rotation of the page.\n\n        This number has to be a multiple of 90 degrees: 0, 90, 180, or 270 are\n        valid values. This property does not affect ``/Contents``.\n        '
        rotate_obj = self.get(PG.ROTATE, 0)
        return rotate_obj if isinstance(rotate_obj, int) else rotate_obj.get_object()

    @rotation.setter
    def rotation(self, r: float) -> None:
        if False:
            print('Hello World!')
        self[NameObject(PG.ROTATE)] = NumberObject((int(r) + 45) // 90 * 90 % 360)

    def transfer_rotation_to_content(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Apply the rotation of the page to the content and the media/crop/...\n        boxes.\n\n        It's recommended to apply this function before page merging.\n        "
        r = -self.rotation
        self.rotation = 0
        mb = RectangleObject(self.mediabox)
        trsf = Transformation().translate(-float(mb.left + mb.width / 2), -float(mb.bottom + mb.height / 2)).rotate(r)
        pt1 = trsf.apply_on(mb.lower_left)
        pt2 = trsf.apply_on(mb.upper_right)
        trsf = trsf.translate(-min(pt1[0], pt2[0]), -min(pt1[1], pt2[1]))
        self.add_transformation(trsf, False)
        for b in ['/MediaBox', '/CropBox', '/BleedBox', '/TrimBox', '/ArtBox']:
            if b in self:
                rr = RectangleObject(self[b])
                pt1 = trsf.apply_on(rr.lower_left)
                pt2 = trsf.apply_on(rr.upper_right)
                self[NameObject(b)] = RectangleObject((min(pt1[0], pt2[0]), min(pt1[1], pt2[1]), max(pt1[0], pt2[0]), max(pt1[1], pt2[1])))

    def rotate(self, angle: int) -> 'PageObject':
        if False:
            print('Hello World!')
        '\n        Rotate a page clockwise by increments of 90 degrees.\n\n        Args:\n            angle: Angle to rotate the page.  Must be an increment of 90 deg.\n\n        Returns:\n            The rotated PageObject\n        '
        if angle % 90 != 0:
            raise ValueError('Rotation angle must be a multiple of 90')
        self[NameObject(PG.ROTATE)] = NumberObject(self.rotation + angle)
        return self

    def rotate_clockwise(self, angle: int) -> 'PageObject':
        if False:
            print('Hello World!')
        deprecation_with_replacement('rotate_clockwise', 'rotate', '3.0.0')
        return self.rotate(angle)

    def rotateClockwise(self, angle: int) -> 'PageObject':
        if False:
            print('Hello World!')
        '\n        Use :meth:`rotate_clockwise` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('rotateClockwise', 'rotate', '3.0.0')
        return self.rotate(angle)

    def rotateCounterClockwise(self, angle: int) -> 'PageObject':
        if False:
            return 10
        '\n        Use :meth:`rotate_clockwise` with a negative argument instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('rotateCounterClockwise', 'rotate', '3.0.0')
        return self.rotate(-angle)

    def _merge_resources(self, res1: DictionaryObject, res2: DictionaryObject, resource: Any, new_res1: bool=True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        try:
            assert isinstance(self.indirect_reference, IndirectObject)
            pdf = self.indirect_reference.pdf
            is_pdf_writer = hasattr(pdf, '_add_object')
        except (AssertionError, AttributeError):
            pdf = None
            is_pdf_writer = False

        def compute_unique_key(base_key: str) -> Tuple[str, bool]:
            if False:
                while True:
                    i = 10
            "\n            Find a key that either doesn't already exist or has the same value\n            (indicated by the bool)\n\n            Args:\n                base_key: An index is added to this to get the computed key\n\n            Returns:\n                A tuple (computed key, bool) where the boolean indicates\n                if there is a resource of the given computed_key with the same\n                value.\n            "
            value = page2res.raw_get(base_key)
            computed_key = base_key
            idx = 0
            while computed_key in new_res:
                if new_res.raw_get(computed_key) == value:
                    return (computed_key, True)
                computed_key = f'{base_key}-{idx}'
                idx += 1
            return (computed_key, False)
        if new_res1:
            new_res = DictionaryObject()
            new_res.update(res1.get(resource, DictionaryObject()).get_object())
        else:
            new_res = cast(DictionaryObject, res1[resource])
        page2res = cast(DictionaryObject, res2.get(resource, DictionaryObject()).get_object())
        rename_res = {}
        for key in page2res:
            (unique_key, same_value) = compute_unique_key(key)
            newname = NameObject(unique_key)
            if key != unique_key:
                rename_res[key] = newname
            if not same_value:
                if is_pdf_writer:
                    new_res[newname] = page2res.raw_get(key).clone(pdf)
                    try:
                        new_res[newname] = new_res[newname].indirect_reference
                    except AttributeError:
                        pass
                else:
                    new_res[newname] = page2res.raw_get(key)
            lst = sorted(new_res.items())
            new_res.clear()
            for el in lst:
                new_res[el[0]] = el[1]
        return (new_res, rename_res)

    @staticmethod
    def _content_stream_rename(stream: ContentStream, rename: Dict[Any, Any], pdf: Union[None, PdfReaderProtocol, PdfWriterProtocol]) -> ContentStream:
        if False:
            for i in range(10):
                print('nop')
        if not rename:
            return stream
        stream = ContentStream(stream, pdf)
        for (operands, _operator) in stream.operations:
            if isinstance(operands, list):
                for (i, op) in enumerate(operands):
                    if isinstance(op, NameObject):
                        operands[i] = rename.get(op, op)
            elif isinstance(operands, dict):
                for (i, op) in operands.items():
                    if isinstance(op, NameObject):
                        operands[i] = rename.get(op, op)
            else:
                raise KeyError(f'type of operands is {type(operands)}')
        return stream

    @staticmethod
    def _add_transformation_matrix(contents: Any, pdf: Union[None, PdfReaderProtocol, PdfWriterProtocol], ctm: CompressedTransformationMatrix) -> ContentStream:
        if False:
            for i in range(10):
                print('nop')
        'Add transformation matrix at the beginning of the given contents stream.'
        (a, b, c, d, e, f) = ctm
        contents = ContentStream(contents, pdf)
        contents.operations.insert(0, [[FloatObject(a), FloatObject(b), FloatObject(c), FloatObject(d), FloatObject(e), FloatObject(f)], ' cm'])
        return contents

    def _get_contents_as_bytes(self) -> Optional[bytes]:
        if False:
            i = 10
            return i + 15
        "\n        Return the page contents as bytes.\n\n        Returns:\n            The ``/Contents`` object as bytes, or ``None`` if it doesn't exist.\n\n        "
        if PG.CONTENTS in self:
            obj = self[PG.CONTENTS].get_object()
            if isinstance(obj, list):
                return b''.join((x.get_object().get_data() for x in obj))
            else:
                return cast(bytes, cast(EncodedStreamObject, obj).get_data())
        else:
            return None

    def get_contents(self) -> Optional[ContentStream]:
        if False:
            while True:
                i = 10
        "\n        Access the page contents.\n\n        Returns:\n            The ``/Contents`` object, or ``None`` if it doesn't exist.\n            ``/Contents`` is optional, as described in PDF Reference  7.7.3.3\n        "
        if PG.CONTENTS in self:
            try:
                pdf = cast(IndirectObject, self.indirect_reference).pdf
            except AttributeError:
                pdf = None
            obj = self[PG.CONTENTS].get_object()
            if isinstance(obj, NullObject):
                return None
            else:
                return ContentStream(obj, pdf)
        else:
            return None

    def getContents(self) -> Optional[ContentStream]:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`get_contents` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('getContents', 'get_contents', '3.0.0')
        return self.get_contents()

    def replace_contents(self, content: Union[None, ContentStream, EncodedStreamObject, ArrayObject]) -> None:
        if False:
            return 10
        '\n        Replace the page contents with the new content and nullify old objects\n        Args:\n            content : new content. if None delete the content field.\n        '
        if not hasattr(self, 'indirect_reference') or self.indirect_reference is None:
            self[NameObject(PG.CONTENTS)] = content
            return
        if isinstance(self.get(PG.CONTENTS, None), ArrayObject):
            for o in self[PG.CONTENTS]:
                try:
                    self._objects[o.indirect_reference.idnum - 1] = NullObject()
                except AttributeError:
                    pass
        if isinstance(content, ArrayObject):
            for i in range(len(content)):
                content[i] = self.indirect_reference.pdf._add_object(content[i])
        if content is None:
            if PG.CONTENTS not in self:
                return
            else:
                assert self.indirect_reference is not None
                assert self[PG.CONTENTS].indirect_reference is not None
                self.indirect_reference.pdf._objects[self[PG.CONTENTS].indirect_reference.idnum - 1] = NullObject()
                del self[PG.CONTENTS]
        elif not hasattr(self.get(PG.CONTENTS, None), 'indirect_reference'):
            try:
                self[NameObject(PG.CONTENTS)] = self.indirect_reference.pdf._add_object(content)
            except AttributeError:
                self[NameObject(PG.CONTENTS)] = content
        else:
            content.indirect_reference = self[PG.CONTENTS].indirect_reference
            try:
                self.indirect_reference.pdf._objects[content.indirect_reference.idnum - 1] = content
            except AttributeError:
                self[NameObject(PG.CONTENTS)] = content

    def merge_page(self, page2: 'PageObject', expand: bool=False, over: bool=True) -> None:
        if False:
            return 10
        '\n        Merge the content streams of two pages into one.\n\n        Resource references\n        (i.e. fonts) are maintained from both pages.  The mediabox/cropbox/etc\n        of this page are not altered.  The parameter page\'s content stream will\n        be added to the end of this page\'s content stream, meaning that it will\n        be drawn after, or "on top" of this page.\n\n        Args:\n            page2: The page to be merged into this one. Should be\n                an instance of :class:`PageObject<PageObject>`.\n            over: set the page2 content over page1 if True(default) else under\n            expand: If true, the current page dimensions will be\n                expanded to accommodate the dimensions of the page to be merged.\n        '
        self._merge_page(page2, over=over, expand=expand)

    def mergePage(self, page2: 'PageObject') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Use :meth:`merge_page` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('mergePage', 'merge_page', '3.0.0')
        return self.merge_page(page2)

    def _merge_page(self, page2: 'PageObject', page2transformation: Optional[Callable[[Any], ContentStream]]=None, ctm: Optional[CompressedTransformationMatrix]=None, over: bool=True, expand: bool=False) -> None:
        if False:
            print('Hello World!')
        try:
            assert isinstance(self.indirect_reference, IndirectObject)
            if hasattr(self.indirect_reference.pdf, '_add_object'):
                return self._merge_page_writer(page2, page2transformation, ctm, over, expand)
        except (AssertionError, AttributeError):
            pass
        new_resources = DictionaryObject()
        rename = {}
        try:
            original_resources = cast(DictionaryObject, self[PG.RESOURCES].get_object())
        except KeyError:
            original_resources = DictionaryObject()
        try:
            page2resources = cast(DictionaryObject, page2[PG.RESOURCES].get_object())
        except KeyError:
            page2resources = DictionaryObject()
        new_annots = ArrayObject()
        for page in (self, page2):
            if PG.ANNOTS in page:
                annots = page[PG.ANNOTS]
                if isinstance(annots, ArrayObject):
                    new_annots.extend(annots)
        for res in (RES.EXT_G_STATE, RES.FONT, RES.XOBJECT, RES.COLOR_SPACE, RES.PATTERN, RES.SHADING, RES.PROPERTIES):
            (new, newrename) = self._merge_resources(original_resources, page2resources, res)
            if new:
                new_resources[NameObject(res)] = new
                rename.update(newrename)
        new_resources[NameObject(RES.PROC_SET)] = ArrayObject(sorted(set(original_resources.get(RES.PROC_SET, ArrayObject()).get_object()).union(set(page2resources.get(RES.PROC_SET, ArrayObject()).get_object()))))
        new_content_array = ArrayObject()
        original_content = self.get_contents()
        if original_content is not None:
            original_content.isolate_graphics_state()
            new_content_array.append(original_content)
        page2content = page2.get_contents()
        if page2content is not None:
            rect = getattr(page2, MERGE_CROP_BOX)
            page2content.operations.insert(0, (map(FloatObject, [rect.left, rect.bottom, rect.width, rect.height]), 're'))
            page2content.operations.insert(1, ([], 'W'))
            page2content.operations.insert(2, ([], 'n'))
            if page2transformation is not None:
                page2content = page2transformation(page2content)
            page2content = PageObject._content_stream_rename(page2content, rename, self.pdf)
            page2content.isolate_graphics_state()
            if over:
                new_content_array.append(page2content)
            else:
                new_content_array.insert(0, page2content)
        if expand:
            self._expand_mediabox(page2, ctm)
        self.replace_contents(ContentStream(new_content_array, self.pdf))
        self[NameObject(PG.RESOURCES)] = new_resources
        self[NameObject(PG.ANNOTS)] = new_annots

    def _merge_page_writer(self, page2: 'PageObject', page2transformation: Optional[Callable[[Any], ContentStream]]=None, ctm: Optional[CompressedTransformationMatrix]=None, over: bool=True, expand: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        assert isinstance(self.indirect_reference, IndirectObject)
        pdf = self.indirect_reference.pdf
        rename = {}
        if PG.RESOURCES not in self:
            self[NameObject(PG.RESOURCES)] = DictionaryObject()
        original_resources = cast(DictionaryObject, self[PG.RESOURCES].get_object())
        if PG.RESOURCES not in page2:
            page2resources = DictionaryObject()
        else:
            page2resources = cast(DictionaryObject, page2[PG.RESOURCES].get_object())
        for res in (RES.EXT_G_STATE, RES.FONT, RES.XOBJECT, RES.COLOR_SPACE, RES.PATTERN, RES.SHADING, RES.PROPERTIES):
            if res in page2resources:
                if res not in original_resources:
                    original_resources[NameObject(res)] = DictionaryObject()
                (_, newrename) = self._merge_resources(original_resources, page2resources, res, False)
                rename.update(newrename)
        if RES.PROC_SET in page2resources:
            if RES.PROC_SET not in original_resources:
                original_resources[NameObject(RES.PROC_SET)] = ArrayObject()
            arr = cast(ArrayObject, original_resources[RES.PROC_SET])
            for x in cast(ArrayObject, page2resources[RES.PROC_SET]):
                if x not in arr:
                    arr.append(x)
            arr.sort()
        if PG.ANNOTS in page2:
            if PG.ANNOTS not in self:
                self[NameObject(PG.ANNOTS)] = ArrayObject()
            annots = cast(ArrayObject, self[PG.ANNOTS].get_object())
            if ctm is None:
                trsf = Transformation()
            else:
                trsf = Transformation(ctm)
            for a in cast(ArrayObject, page2[PG.ANNOTS]):
                a = a.get_object()
                aa = a.clone(pdf, ignore_fields=('/P', '/StructParent', '/Parent'), force_duplicate=True)
                r = cast(ArrayObject, a['/Rect'])
                pt1 = trsf.apply_on((r[0], r[1]), True)
                pt2 = trsf.apply_on((r[2], r[3]), True)
                aa[NameObject('/Rect')] = ArrayObject((min(pt1[0], pt2[0]), min(pt1[1], pt2[1]), max(pt1[0], pt2[0]), max(pt1[1], pt2[1])))
                if '/QuadPoints' in a:
                    q = cast(ArrayObject, a['/QuadPoints'])
                    aa[NameObject('/QuadPoints')] = ArrayObject(trsf.apply_on((q[0], q[1]), True) + trsf.apply_on((q[2], q[3]), True) + trsf.apply_on((q[4], q[5]), True) + trsf.apply_on((q[6], q[7]), True))
                try:
                    aa['/Popup'][NameObject('/Parent')] = aa.indirect_reference
                except KeyError:
                    pass
                try:
                    aa[NameObject('/P')] = self.indirect_reference
                    annots.append(aa.indirect_reference)
                except AttributeError:
                    pass
        new_content_array = ArrayObject()
        original_content = self.get_contents()
        if original_content is not None:
            original_content.isolate_graphics_state()
            new_content_array.append(original_content)
        page2content = page2.get_contents()
        if page2content is not None:
            rect = getattr(page2, MERGE_CROP_BOX)
            page2content.operations.insert(0, (map(FloatObject, [rect.left, rect.bottom, rect.width, rect.height]), 're'))
            page2content.operations.insert(1, ([], 'W'))
            page2content.operations.insert(2, ([], 'n'))
            if page2transformation is not None:
                page2content = page2transformation(page2content)
            page2content = PageObject._content_stream_rename(page2content, rename, self.pdf)
            page2content.isolate_graphics_state()
            if over:
                new_content_array.append(page2content)
            else:
                new_content_array.insert(0, page2content)
        if expand:
            self._expand_mediabox(page2, ctm)
        self.replace_contents(new_content_array)

    def _expand_mediabox(self, page2: 'PageObject', ctm: Optional[CompressedTransformationMatrix]) -> None:
        if False:
            print('Hello World!')
        corners1 = (self.mediabox.left.as_numeric(), self.mediabox.bottom.as_numeric(), self.mediabox.right.as_numeric(), self.mediabox.top.as_numeric())
        corners2 = (page2.mediabox.left.as_numeric(), page2.mediabox.bottom.as_numeric(), page2.mediabox.left.as_numeric(), page2.mediabox.top.as_numeric(), page2.mediabox.right.as_numeric(), page2.mediabox.top.as_numeric(), page2.mediabox.right.as_numeric(), page2.mediabox.bottom.as_numeric())
        if ctm is not None:
            ctm = tuple((float(x) for x in ctm))
            new_x = tuple((ctm[0] * corners2[i] + ctm[2] * corners2[i + 1] + ctm[4] for i in range(0, 8, 2)))
            new_y = tuple((ctm[1] * corners2[i] + ctm[3] * corners2[i + 1] + ctm[5] for i in range(0, 8, 2)))
        else:
            new_x = corners2[0:8:2]
            new_y = corners2[1:8:2]
        lowerleft = (min(new_x), min(new_y))
        upperright = (max(new_x), max(new_y))
        lowerleft = (min(corners1[0], lowerleft[0]), min(corners1[1], lowerleft[1]))
        upperright = (max(corners1[2], upperright[0]), max(corners1[3], upperright[1]))
        self.mediabox.lower_left = lowerleft
        self.mediabox.upper_right = upperright

    def merge_transformed_page(self, page2: 'PageObject', ctm: Union[CompressedTransformationMatrix, Transformation], over: bool=True, expand: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        merge_transformed_page is similar to merge_page, but a transformation\n        matrix is applied to the merged stream.\n\n        Args:\n          page2: The page to be merged into this one.\n          ctm: a 6-element tuple containing the operands of the\n                 transformation matrix\n          over: set the page2 content over page1 if True(default) else under\n          expand: Whether the page should be expanded to fit the dimensions\n            of the page to be merged.\n        '
        if isinstance(ctm, Transformation):
            ctm = ctm.ctm
        self._merge_page(page2, lambda page2Content: PageObject._add_transformation_matrix(page2Content, page2.pdf, cast(CompressedTransformationMatrix, ctm)), ctm, over, expand)

    def mergeTransformedPage(self, page2: 'PageObject', ctm: Union[CompressedTransformationMatrix, Transformation], expand: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        deprecated\n\n        deprecated:: 1.28.0\n\n            Use :meth:`merge_transformed_page`  instead.\n        '
        deprecation_with_replacement('page.mergeTransformedPage(page2, ctm,expand)', 'page.merge_transformed_page(page2,ctm,expand)', '3.0.0')
        self.merge_transformed_page(page2, ctm, expand)

    def merge_scaled_page(self, page2: 'PageObject', scale: float, over: bool=True, expand: bool=False) -> None:
        if False:
            return 10
        '\n        merge_scaled_page is similar to merge_page, but the stream to be merged\n        is scaled by applying a transformation matrix.\n\n        Args:\n          page2: The page to be merged into this one.\n          scale: The scaling factor\n          over: set the page2 content over page1 if True(default) else under\n          expand: Whether the page should be expanded to fit the\n            dimensions of the page to be merged.\n        '
        op = Transformation().scale(scale, scale)
        self.merge_transformed_page(page2, op, over, expand)

    def mergeScaledPage(self, page2: 'PageObject', scale: float, expand: bool=False) -> None:
        if False:
            return 10
        '\n        deprecated\n\n        .. deprecated:: 1.28.0\n\n            Use :meth:`merge_scaled_page` instead.\n        '
        deprecation_with_replacement('page.mergeScaledPage(page2, scale, expand)', 'page2.merge_scaled_page(page2, scale, expand)', '3.0.0')
        self.merge_scaled_page(page2, scale, expand)

    def merge_rotated_page(self, page2: 'PageObject', rotation: float, over: bool=True, expand: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        merge_rotated_page is similar to merge_page, but the stream to be merged\n        is rotated by applying a transformation matrix.\n\n        Args:\n          page2: The page to be merged into this one.\n          rotation: The angle of the rotation, in degrees\n          over: set the page2 content over page1 if True(default) else under\n          expand: Whether the page should be expanded to fit the\n            dimensions of the page to be merged.\n        '
        op = Transformation().rotate(rotation)
        self.merge_transformed_page(page2, op, over, expand)

    def mergeRotatedPage(self, page2: 'PageObject', rotation: float, expand: bool=False) -> None:
        if False:
            return 10
        '\n        deprecated\n\n        .. deprecated:: 1.28.0\n\n            Use :meth:`add_transformation` and :meth:`merge_page` instead.\n        '
        deprecation_with_replacement('page.mergeRotatedPage(page2, rotation, expand)', 'page2.mergeotatedPage(page2, rotation, expand)', '3.0.0')
        self.merge_rotated_page(page2, rotation, expand)

    def merge_translated_page(self, page2: 'PageObject', tx: float, ty: float, over: bool=True, expand: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        mergeTranslatedPage is similar to merge_page, but the stream to be\n        merged is translated by applying a transformation matrix.\n\n        Args:\n          page2: the page to be merged into this one.\n          tx: The translation on X axis\n          ty: The translation on Y axis\n          over: set the page2 content over page1 if True(default) else under\n          expand: Whether the page should be expanded to fit the\n            dimensions of the page to be merged.\n        '
        op = Transformation().translate(tx, ty)
        self.merge_transformed_page(page2, op, over, expand)

    def mergeTranslatedPage(self, page2: 'PageObject', tx: float, ty: float, expand: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        deprecated\n\n        .. deprecated:: 1.28.0\n\n            Use :meth:`merge_translated_page` instead.\n        '
        deprecation_with_replacement('page.mergeTranslatedPage(page2, tx, ty, expand)', 'page2.add_transformation(Transformation().translate(tx, ty)); page.merge_page(page2, expand)', '3.0.0')
        self.merge_translated_page(page2, tx, ty, expand)

    def mergeRotatedTranslatedPage(self, page2: 'PageObject', rotation: float, tx: float, ty: float, expand: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        .. deprecated:: 1.28.0\n\n            Use :meth:`merge_transformed_page` instead.\n        '
        deprecation_with_replacement('page.mergeRotatedTranslatedPage(page2, rotation, tx, ty, expand)', 'page.merge_transformed_page(page2, Transformation().rotate(rotation).translate(tx, ty), expand);', '3.0.0')
        op = Transformation().translate(-tx, -ty).rotate(rotation).translate(tx, ty)
        return self.merge_transformed_page(page2, op, expand)

    def mergeRotatedScaledPage(self, page2: 'PageObject', rotation: float, scale: float, expand: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        .. deprecated:: 1.28.0\n\n            Use :meth:`merge_transformed_page` instead.\n        '
        deprecation_with_replacement('page.mergeRotatedScaledPage(page2, rotation, scale, expand)', 'page.merge_transformed_page(page2, Transformation().rotate(rotation).scale(scale)); page.merge_page(page2, expand)', '3.0.0')
        op = Transformation().rotate(rotation).scale(scale, scale)
        self.mergeTransformedPage(page2, op, expand)

    def mergeScaledTranslatedPage(self, page2: 'PageObject', scale: float, tx: float, ty: float, expand: bool=False) -> None:
        if False:
            return 10
        '\n        mergeScaledTranslatedPage is similar to merge_page, but the stream to\n        be merged is translated and scaled by applying a transformation matrix.\n\n        :param PageObject page2: the page to be merged into this one. Should be\n            an instance of :class:`PageObject<PageObject>`.\n        :param float scale: The scaling factor\n        :param float tx: The translation on X axis\n        :param float ty: The translation on Y axis\n        :param bool expand: Whether the page should be expanded to fit the\n            dimensions of the page to be merged.\n\n        .. deprecated:: 1.28.0\n\n            Use :meth:`add_transformation` and :meth:`merge_page` instead.\n        '
        deprecation_with_replacement('page.mergeScaledTranslatedPage(page2, scale, tx, ty, expand)', 'page2.add_transformation(Transformation().scale(scale).translate(tx, ty)); page.merge_page(page2, expand)', '3.0.0')
        op = Transformation().scale(scale, scale).translate(tx, ty)
        return self.mergeTransformedPage(page2, op, expand)

    def mergeRotatedScaledTranslatedPage(self, page2: 'PageObject', rotation: float, scale: float, tx: float, ty: float, expand: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        mergeRotatedScaledTranslatedPage is similar to merge_page, but the\n        stream to be merged is translated, rotated and scaled by applying a\n        transformation matrix.\n\n        :param PageObject page2: the page to be merged into this one. Should be\n            an instance of :class:`PageObject<PageObject>`.\n        :param float tx: The translation on X axis\n        :param float ty: The translation on Y axis\n        :param float rotation: The angle of the rotation, in degrees\n        :param float scale: The scaling factor\n        :param bool expand: Whether the page should be expanded to fit the\n            dimensions of the page to be merged.\n\n        .. deprecated:: 1.28.0\n\n            Use :meth:`add_transformation` and :meth:`merge_page` instead.\n        '
        deprecation_with_replacement('page.mergeRotatedScaledTranslatedPage(page2, rotation, tx, ty, expand)', 'page2.add_transformation(Transformation().rotate(rotation).scale(scale)); page.merge_page(page2, expand)', '3.0.0')
        op = Transformation().rotate(rotation).scale(scale, scale).translate(tx, ty)
        self.mergeTransformedPage(page2, op, expand)

    def add_transformation(self, ctm: Union[Transformation, CompressedTransformationMatrix], expand: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Apply a transformation matrix to the page.\n\n        Args:\n            ctm: A 6-element tuple containing the operands of the\n                transformation matrix. Alternatively, a\n                :py:class:`Transformation<pypdf.Transformation>`\n                object can be passed.\n\n        See :doc:`/user/cropping-and-transforming`.\n        '
        if isinstance(ctm, Transformation):
            ctm = ctm.ctm
        content = self.get_contents()
        if content is not None:
            content = PageObject._add_transformation_matrix(content, self.pdf, ctm)
            content.isolate_graphics_state()
            self.replace_contents(content)
        if expand:
            corners = [self.mediabox.left.as_numeric(), self.mediabox.bottom.as_numeric(), self.mediabox.left.as_numeric(), self.mediabox.top.as_numeric(), self.mediabox.right.as_numeric(), self.mediabox.top.as_numeric(), self.mediabox.right.as_numeric(), self.mediabox.bottom.as_numeric()]
            ctm = tuple((float(x) for x in ctm))
            new_x = [ctm[0] * corners[i] + ctm[2] * corners[i + 1] + ctm[4] for i in range(0, 8, 2)]
            new_y = [ctm[1] * corners[i] + ctm[3] * corners[i + 1] + ctm[5] for i in range(0, 8, 2)]
            lowerleft = (min(new_x), min(new_y))
            upperright = (max(new_x), max(new_y))
            self.mediabox.lower_left = lowerleft
            self.mediabox.upper_right = upperright

    def addTransformation(self, ctm: CompressedTransformationMatrix) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`add_transformation` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('addTransformation', 'add_transformation', '3.0.0')
        self.add_transformation(ctm)

    def scale(self, sx: float, sy: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Scale a page by the given factors by applying a transformation matrix\n        to its content and updating the page size.\n\n        This updates the mediabox, the cropbox, and the contents\n        of the page.\n\n        Args:\n            sx: The scaling factor on horizontal axis.\n            sy: The scaling factor on vertical axis.\n        '
        self.add_transformation((sx, 0, 0, sy, 0, 0))
        self.cropbox = self.cropbox.scale(sx, sy)
        self.artbox = self.artbox.scale(sx, sy)
        self.bleedbox = self.bleedbox.scale(sx, sy)
        self.trimbox = self.trimbox.scale(sx, sy)
        self.mediabox = self.mediabox.scale(sx, sy)
        if PG.ANNOTS in self:
            annotations = self[PG.ANNOTS]
            if isinstance(annotations, ArrayObject):
                for annotation in annotations:
                    annotation_obj = annotation.get_object()
                    if ADA.Rect in annotation_obj:
                        rectangle = annotation_obj[ADA.Rect]
                        if isinstance(rectangle, ArrayObject):
                            rectangle[0] = FloatObject(float(rectangle[0]) * sx)
                            rectangle[1] = FloatObject(float(rectangle[1]) * sy)
                            rectangle[2] = FloatObject(float(rectangle[2]) * sx)
                            rectangle[3] = FloatObject(float(rectangle[3]) * sy)
        if PG.VP in self:
            viewport = self[PG.VP]
            if isinstance(viewport, ArrayObject):
                bbox = viewport[0]['/BBox']
            else:
                bbox = viewport['/BBox']
            scaled_bbox = RectangleObject((float(bbox[0]) * sx, float(bbox[1]) * sy, float(bbox[2]) * sx, float(bbox[3]) * sy))
            if isinstance(viewport, ArrayObject):
                self[NameObject(PG.VP)][NumberObject(0)][NameObject('/BBox')] = scaled_bbox
            else:
                self[NameObject(PG.VP)][NameObject('/BBox')] = scaled_bbox

    def scale_by(self, factor: float) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Scale a page by the given factor by applying a transformation matrix to\n        its content and updating the page size.\n\n        Args:\n            factor: The scaling factor (for both X and Y axis).\n        '
        self.scale(factor, factor)

    def scaleBy(self, factor: float) -> None:
        if False:
            return 10
        '\n        Use :meth:`scale_by` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('scaleBy', 'scale_by', '3.0.0')
        self.scale(factor, factor)

    def scale_to(self, width: float, height: float) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Scale a page to the specified dimensions by applying a transformation\n        matrix to its content and updating the page size.\n\n        Args:\n            width: The new width.\n            height: The new height.\n        '
        sx = width / float(self.mediabox.width)
        sy = height / float(self.mediabox.height)
        self.scale(sx, sy)

    def scaleTo(self, width: float, height: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :meth:`scale_to` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('scaleTo', 'scale_to', '3.0.0')
        self.scale_to(width, height)

    def compress_content_streams(self, level: int=-1) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compress the size of this page by joining all content streams and\n        applying a FlateDecode filter.\n\n        However, it is possible that this function will perform no action if\n        content stream compression becomes "automatic".\n        '
        content = self.get_contents()
        if content is not None:
            content_obj = content.flate_encode(level)
            try:
                content.indirect_reference.pdf._objects[content.indirect_reference.idnum - 1] = content_obj
            except AttributeError:
                if self.indirect_reference is not None and hasattr(self.indirect_reference.pdf, '_add_object'):
                    self.replace_contents(content_obj)
                else:
                    raise ValueError('Page must be part of a PdfWriter')

    def compressContentStreams(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Use :meth:`compress_content_streams` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('compressContentStreams', 'compress_content_streams', '3.0.0')
        self.compress_content_streams()

    @property
    def page_number(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Read-only property which return the page number with the pdf file.\n\n        Returns:\n            int : page number ; -1 if the page is not attached to a pdf\n        '
        if self.indirect_reference is None:
            return -1
        else:
            try:
                lst = self.indirect_reference.pdf.pages
                return lst.index(self)
            except ValueError:
                return -1

    def _debug_for_extract(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        out = ''
        for (ope, op) in ContentStream(self['/Contents'].get_object(), self.pdf, 'bytes').operations:
            if op == b'TJ':
                s = [x for x in ope[0] if isinstance(x, str)]
            else:
                s = []
            out += op.decode('utf-8') + ' ' + ''.join(s) + ope.__repr__() + '\n'
        out += '\n=============================\n'
        try:
            for fo in self[PG.RESOURCES]['/Font']:
                out += fo + '\n'
                out += self[PG.RESOURCES]['/Font'][fo].__repr__() + '\n'
                try:
                    enc_repr = self[PG.RESOURCES]['/Font'][fo]['/Encoding'].__repr__()
                    out += enc_repr + '\n'
                except Exception:
                    pass
                try:
                    out += self[PG.RESOURCES]['/Font'][fo]['/ToUnicode'].get_data().decode() + '\n'
                except Exception:
                    pass
        except KeyError:
            out += 'No Font\n'
        return out

    def _extract_text(self, obj: Any, pdf: Any, orientations: Tuple[int, ...]=(0, 90, 180, 270), space_width: float=200.0, content_key: Optional[str]=PG.CONTENTS, visitor_operand_before: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_operand_after: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        See extract_text for most arguments.\n\n        Args:\n            content_key: indicate the default key where to extract data\n                None = the object; this allow to reuse the function on XObject\n                default = "/Content"\n        '
        text: str = ''
        output: str = ''
        rtl_dir: bool = False
        cmaps: Dict[str, Tuple[str, float, Union[str, Dict[int, str]], Dict[str, str], DictionaryObject]] = {}
        try:
            objr = obj
            while NameObject(PG.RESOURCES) not in objr:
                objr = objr['/Parent'].get_object()
            resources_dict = cast(DictionaryObject, objr[PG.RESOURCES])
        except Exception:
            return ''
        if '/Font' in resources_dict:
            for f in cast(DictionaryObject, resources_dict['/Font']):
                cmaps[f] = build_char_map(f, space_width, obj)
        cmap: Tuple[Union[str, Dict[int, str]], Dict[str, str], str, Optional[DictionaryObject]] = ('charmap', {}, 'NotInitialized', None)
        try:
            content = obj[content_key].get_object() if isinstance(content_key, str) else obj
            if not isinstance(content, ContentStream):
                content = ContentStream(content, pdf, 'bytes')
        except KeyError:
            return ''
        cm_matrix: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        cm_stack = []
        tm_matrix: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        cm_prev: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        tm_prev: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        memo_cm: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        memo_tm: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        char_scale = 1.0
        space_scale = 1.0
        _space_width: float = 500.0
        TL = 0.0
        font_size = 12.0

        def current_spacewidth() -> float:
            if False:
                i = 10
                return i + 15
            return _space_width / 1000.0

        def process_operation(operator: bytes, operands: List[Any]) -> None:
            if False:
                return 10
            nonlocal cm_matrix, cm_stack, tm_matrix, cm_prev, tm_prev, memo_cm, memo_tm
            nonlocal char_scale, space_scale, _space_width, TL, font_size, cmap
            nonlocal orientations, rtl_dir, visitor_text, output, text
            global CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS
            check_crlf_space: bool = False
            if operator == b'BT':
                tm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                output += text
                if visitor_text is not None:
                    visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
                text = ''
                memo_cm = cm_matrix.copy()
                memo_tm = tm_matrix.copy()
                return None
            elif operator == b'ET':
                output += text
                if visitor_text is not None:
                    visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
                text = ''
                memo_cm = cm_matrix.copy()
                memo_tm = tm_matrix.copy()
            elif operator == b'q':
                cm_stack.append((cm_matrix, cmap, font_size, char_scale, space_scale, _space_width, TL))
            elif operator == b'Q':
                try:
                    (cm_matrix, cmap, font_size, char_scale, space_scale, _space_width, TL) = cm_stack.pop()
                except Exception:
                    cm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif operator == b'cm':
                output += text
                if visitor_text is not None:
                    visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
                text = ''
                cm_matrix = mult([float(operands[0]), float(operands[1]), float(operands[2]), float(operands[3]), float(operands[4]), float(operands[5])], cm_matrix)
                memo_cm = cm_matrix.copy()
                memo_tm = tm_matrix.copy()
            elif operator == b'Tz':
                char_scale = float(operands[0]) / 100.0
            elif operator == b'Tw':
                space_scale = 1.0 + float(operands[0])
            elif operator == b'TL':
                TL = float(operands[0])
            elif operator == b'Tf':
                if text != '':
                    output += text
                    if visitor_text is not None:
                        visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
                text = ''
                memo_cm = cm_matrix.copy()
                memo_tm = tm_matrix.copy()
                try:
                    charMapTuple = cmaps[operands[0]]
                    _space_width = charMapTuple[1]
                    cmap = (charMapTuple[2], charMapTuple[3], operands[0], charMapTuple[4])
                except KeyError:
                    _space_width = unknown_char_map[1]
                    cmap = (unknown_char_map[2], unknown_char_map[3], '???' + operands[0], None)
                try:
                    font_size = float(operands[1])
                except Exception:
                    pass
            elif operator == b'Td':
                check_crlf_space = True
                tx = float(operands[0])
                ty = float(operands[1])
                tm_matrix[4] += tx * tm_matrix[0] + ty * tm_matrix[2]
                tm_matrix[5] += tx * tm_matrix[1] + ty * tm_matrix[3]
            elif operator == b'Tm':
                check_crlf_space = True
                tm_matrix = [float(operands[0]), float(operands[1]), float(operands[2]), float(operands[3]), float(operands[4]), float(operands[5])]
            elif operator == b'T*':
                check_crlf_space = True
                tm_matrix[5] -= TL
            elif operator == b'Tj':
                check_crlf_space = True
                (text, rtl_dir) = handle_tj(text, operands, cm_matrix, tm_matrix, cmap, orientations, output, font_size, rtl_dir, visitor_text)
            else:
                return None
            if check_crlf_space:
                try:
                    (text, output, cm_prev, tm_prev) = crlf_space_check(text, (cm_prev, tm_prev), (cm_matrix, tm_matrix), (memo_cm, memo_tm), cmap, orientations, output, font_size, visitor_text, current_spacewidth())
                    if text == '':
                        memo_cm = cm_matrix.copy()
                        memo_tm = tm_matrix.copy()
                except OrientationNotFoundError:
                    return None
        for (operands, operator) in content.operations:
            if visitor_operand_before is not None:
                visitor_operand_before(operator, operands, cm_matrix, tm_matrix)
            if operator == b"'":
                process_operation(b'T*', [])
                process_operation(b'Tj', operands)
            elif operator == b'"':
                process_operation(b'Tw', [operands[0]])
                process_operation(b'Tc', [operands[1]])
                process_operation(b'T*', [])
                process_operation(b'Tj', operands[2:])
            elif operator == b'TD':
                process_operation(b'TL', [-operands[1]])
                process_operation(b'Td', operands)
            elif operator == b'TJ':
                for op in operands[0]:
                    if isinstance(op, (str, bytes)):
                        process_operation(b'Tj', [op])
                    if isinstance(op, (int, float, NumberObject, FloatObject)) and (abs(float(op)) >= _space_width and len(text) > 0 and (text[-1] != ' ')):
                        process_operation(b'Tj', [' '])
            elif operator == b'Do':
                output += text
                if visitor_text is not None:
                    visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
                try:
                    if output[-1] != '\n':
                        output += '\n'
                        if visitor_text is not None:
                            visitor_text('\n', memo_cm, memo_tm, cmap[3], font_size)
                except IndexError:
                    pass
                try:
                    xobj = resources_dict['/XObject']
                    if xobj[operands[0]]['/Subtype'] != '/Image':
                        text = self.extract_xform_text(xobj[operands[0]], orientations, space_width, visitor_operand_before, visitor_operand_after, visitor_text)
                        output += text
                        if visitor_text is not None:
                            visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
                except Exception:
                    logger_warning(f' impossible to decode XFormObject {operands[0]}', __name__)
                finally:
                    text = ''
                    memo_cm = cm_matrix.copy()
                    memo_tm = tm_matrix.copy()
            else:
                process_operation(operator, operands)
            if visitor_operand_after is not None:
                visitor_operand_after(operator, operands, cm_matrix, tm_matrix)
        output += text
        if text != '' and visitor_text is not None:
            visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
        return output

    def extract_text(self, *args: Any, Tj_sep: Optional[str]=None, TJ_sep: Optional[str]=None, orientations: Union[int, Tuple[int, ...]]=(0, 90, 180, 270), space_width: float=200.0, visitor_operand_before: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_operand_after: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]]=None) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Locate all text drawing commands, in the order they are provided in the\n        content stream, and extract the text.\n\n        This works well for some PDF files, but poorly for others, depending on\n        the generator used. This will be refined in the future.\n\n        Do not rely on the order of text coming out of this function, as it\n        will change if this function is made more sophisticated.\n\n        Arabic, Hebrew,... are extracted in the good order.\n        If required an custom RTL range of characters can be defined;\n        see function set_custom_rtl\n\n        Additionally you can provide visitor-methods to get informed on all\n        operations and all text-objects.\n        For example in some PDF files this can be useful to parse tables.\n\n        Args:\n            Tj_sep: Deprecated. Kept for compatibility until pypdf 4.0.0\n            TJ_sep: Deprecated. Kept for compatibility until pypdf 4.0.0\n            orientations: list of orientations text_extraction will look for\n                default = (0, 90, 180, 270)\n                note: currently only 0(Up),90(turned Left), 180(upside Down),\n                270 (turned Right)\n            space_width: force default space width\n                if not extracted from font (default: 200)\n            visitor_operand_before: function to be called before processing an operation.\n                It has four arguments: operator, operand-arguments,\n                current transformation matrix and text matrix.\n            visitor_operand_after: function to be called after processing an operation.\n                It has four arguments: operator, operand-arguments,\n                current transformation matrix and text matrix.\n            visitor_text: function to be called when extracting some text at some position.\n                It has five arguments: text, current transformation matrix,\n                text matrix, font-dictionary and font-size.\n                The font-dictionary may be None in case of unknown fonts.\n                If not None it may e.g. contain key "/BaseFont" with value "/Arial,Bold".\n\n        Returns:\n            The extracted text\n        '
        if len(args) >= 1:
            if isinstance(args[0], str):
                Tj_sep = args[0]
                if len(args) >= 2:
                    if isinstance(args[1], str):
                        TJ_sep = args[1]
                    else:
                        raise TypeError(f'Invalid positional parameter {args[1]}')
                if len(args) >= 3:
                    if isinstance(args[2], (tuple, int)):
                        orientations = args[2]
                    else:
                        raise TypeError(f'Invalid positional parameter {args[2]}')
                if len(args) >= 4:
                    if isinstance(args[3], (float, int)):
                        space_width = args[3]
                    else:
                        raise TypeError(f'Invalid positional parameter {args[3]}')
            elif isinstance(args[0], (tuple, int)):
                orientations = args[0]
                if len(args) >= 2:
                    if isinstance(args[1], (float, int)):
                        space_width = args[1]
                    else:
                        raise TypeError(f'Invalid positional parameter {args[1]}')
            else:
                raise TypeError(f'Invalid positional parameter {args[0]}')
        if Tj_sep is not None or TJ_sep is not None:
            warnings.warn('parameters Tj_Sep, TJ_sep depreciated, and will be removed in pypdf 4.0.0.', DeprecationWarning)
        if isinstance(orientations, int):
            orientations = (orientations,)
        return self._extract_text(self, self.pdf, orientations, space_width, PG.CONTENTS, visitor_operand_before, visitor_operand_after, visitor_text)

    def extract_xform_text(self, xform: EncodedStreamObject, orientations: Tuple[int, ...]=(0, 90, 270, 360), space_width: float=200.0, visitor_operand_before: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_operand_after: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]]=None) -> str:
        if False:
            print('Hello World!')
        '\n        Extract text from an XObject.\n\n        Args:\n            xform:\n            orientations:\n            space_width:  force default space width (if not extracted from font (default 200)\n            visitor_operand_before:\n            visitor_operand_after:\n            visitor_text:\n\n        Returns:\n            The extracted text\n        '
        return self._extract_text(xform, self.pdf, orientations, space_width, None, visitor_operand_before, visitor_operand_after, visitor_text)

    def extractText(self, Tj_sep: str='', TJ_sep: str='') -> str:
        if False:
            print('Hello World!')
        '\n        Use :meth:`extract_text` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('extractText', 'extract_text', '3.0.0')
        return self.extract_text()

    def _get_fonts(self) -> Tuple[Set[str], Set[str]]:
        if False:
            print('Hello World!')
        '\n        Get the names of embedded fonts and unembedded fonts.\n\n        Returns:\n            A tuple (Set of embedded fonts, set of unembedded fonts)\n        '
        obj = self.get_object()
        assert isinstance(obj, DictionaryObject)
        fonts: Set[str] = set()
        embedded: Set[str] = set()
        (fonts, embedded) = _get_fonts_walk(obj, fonts, embedded)
        unembedded = fonts - embedded
        return (embedded, unembedded)
    mediabox = _create_rectangle_accessor(PG.MEDIABOX, ())
    'A :class:`RectangleObject<pypdf.generic.RectangleObject>`, expressed in\n    default user space units, defining the boundaries of the physical medium on\n    which the page is intended to be displayed or printed.'

    @property
    def mediaBox(self) -> RectangleObject:
        if False:
            i = 10
            return i + 15
        '\n        Use :py:attr:`mediabox` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('mediaBox', 'mediabox', '3.0.0')
        return self.mediabox

    @mediaBox.setter
    def mediaBox(self, value: RectangleObject) -> None:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`mediabox` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('mediaBox', 'mediabox', '3.0.0')
        self.mediabox = value
    cropbox = _create_rectangle_accessor('/CropBox', (PG.MEDIABOX,))
    '\n    A :class:`RectangleObject<pypdf.generic.RectangleObject>`, expressed in\n    default user space units, defining the visible region of default user\n    space.\n\n    When the page is displayed or printed, its contents are to be clipped\n    (cropped) to this rectangle and then imposed on the output medium in some\n    implementation-defined manner.  Default value: same as\n    :attr:`mediabox<mediabox>`.\n    '

    @property
    def cropBox(self) -> RectangleObject:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`cropbox` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('cropBox', 'cropbox', '3.0.0')
        return self.cropbox

    @cropBox.setter
    def cropBox(self, value: RectangleObject) -> None:
        if False:
            while True:
                i = 10
        deprecation_with_replacement('cropBox', 'cropbox', '3.0.0')
        self.cropbox = value
    bleedbox = _create_rectangle_accessor('/BleedBox', ('/CropBox', PG.MEDIABOX))
    'A :class:`RectangleObject<pypdf.generic.RectangleObject>`, expressed in\n    default user space units, defining the region to which the contents of the\n    page should be clipped when output in a production environment.'

    @property
    def bleedBox(self) -> RectangleObject:
        if False:
            return 10
        '\n        Use :py:attr:`bleedbox` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('bleedBox', 'bleedbox', '3.0.0')
        return self.bleedbox

    @bleedBox.setter
    def bleedBox(self, value: RectangleObject) -> None:
        if False:
            return 10
        deprecation_with_replacement('bleedBox', 'bleedbox', '3.0.0')
        self.bleedbox = value
    trimbox = _create_rectangle_accessor('/TrimBox', ('/CropBox', PG.MEDIABOX))
    'A :class:`RectangleObject<pypdf.generic.RectangleObject>`, expressed in\n    default user space units, defining the intended dimensions of the finished\n    page after trimming.'

    @property
    def trimBox(self) -> RectangleObject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Use :py:attr:`trimbox` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('trimBox', 'trimbox', '3.0.0')
        return self.trimbox

    @trimBox.setter
    def trimBox(self, value: RectangleObject) -> None:
        if False:
            while True:
                i = 10
        deprecation_with_replacement('trimBox', 'trimbox', '3.0.0')
        self.trimbox = value
    artbox = _create_rectangle_accessor('/ArtBox', ('/CropBox', PG.MEDIABOX))
    "A :class:`RectangleObject<pypdf.generic.RectangleObject>`, expressed in\n    default user space units, defining the extent of the page's meaningful\n    content as intended by the page's creator."

    @property
    def artBox(self) -> RectangleObject:
        if False:
            print('Hello World!')
        '\n        Use :py:attr:`artbox` instead.\n\n        .. deprecated:: 1.28.0\n        '
        deprecation_with_replacement('artBox', 'artbox', '3.0.0')
        return self.artbox

    @artBox.setter
    def artBox(self, value: RectangleObject) -> None:
        if False:
            for i in range(10):
                print('nop')
        deprecation_with_replacement('artBox', 'artbox', '3.0.0')
        self.artbox = value

    @property
    def annotations(self) -> Optional[ArrayObject]:
        if False:
            return 10
        if '/Annots' not in self:
            return None
        else:
            return cast(ArrayObject, self['/Annots'])

    @annotations.setter
    def annotations(self, value: Optional[ArrayObject]) -> None:
        if False:
            print('Hello World!')
        "\n        Set the annotations array of the page.\n\n        Typically you don't want to set this value, but append to it.\n        If you append to it, don't forget to add the object first to the writer\n        and only add the indirect object.\n        "
        if value is None:
            del self[NameObject('/Annots')]
        else:
            self[NameObject('/Annots')] = value

class _VirtualList(Sequence[PageObject]):

    def __init__(self, length_function: Callable[[], int], get_function: Callable[[int], PageObject]) -> None:
        if False:
            print('Hello World!')
        self.length_function = length_function
        self.get_function = get_function
        self.current = -1

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.length_function()

    @overload
    def __getitem__(self, index: int) -> PageObject:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PageObject]:
        if False:
            while True:
                i = 10
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[PageObject, Sequence[PageObject]]:
        if False:
            print('Hello World!')
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            cls = type(self)
            return cls(indices.__len__, lambda idx: self[indices[idx]])
        if not isinstance(index, int):
            raise TypeError('sequence indices must be integers')
        len_self = len(self)
        if index < 0:
            index = len_self + index
        if index < 0 or index >= len_self:
            raise IndexError('sequence index out of range')
        return self.get_function(index)

    def __delitem__(self, index: Union[int, slice]) -> None:
        if False:
            return 10
        if isinstance(index, slice):
            r = list(range(*index.indices(len(self))))
            r.sort()
            r.reverse()
            for p in r:
                del self[p]
            return
        if not isinstance(index, int):
            raise TypeError('index must be integers')
        len_self = len(self)
        if index < 0:
            index = len_self + index
        if index < 0 or index >= len_self:
            raise IndexError('index out of range')
        ind = self[index].indirect_reference
        assert ind is not None
        parent = cast(DictionaryObject, ind.get_object()).get('/Parent', None)
        while parent is not None:
            parent = cast(DictionaryObject, parent.get_object())
            try:
                i = parent['/Kids'].index(ind)
                del parent['/Kids'][i]
                try:
                    assert ind is not None
                    del ind.pdf.flattened_pages[index]
                except AttributeError:
                    pass
                if '/Count' in parent:
                    parent[NameObject('/Count')] = NumberObject(parent['/Count'] - 1)
                if len(parent['/Kids']) == 0:
                    ind = parent.indirect_reference
                    parent = cast(DictionaryObject, parent.get('/Parent', None))
                else:
                    parent = None
            except ValueError:
                raise PdfReadError(f'Page Not Found in Page Tree {ind}')

    def __iter__(self) -> Iterator[PageObject]:
        if False:
            print('Hello World!')
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        p = [f'PageObject({i})' for i in range(self.length_function())]
        return f"[{', '.join(p)}]"

def _get_fonts_walk(obj: DictionaryObject, fnt: Set[str], emb: Set[str]) -> Tuple[Set[str], Set[str]]:
    if False:
        while True:
            i = 10
    "\n    Get the set of all fonts and all embedded fonts.\n\n    Args:\n        obj: Page resources dictionary\n        fnt: font\n        emb: embedded fonts\n\n    Returns:\n        A tuple (fnt, emb)\n\n    If there is a key called 'BaseFont', that is a font that is used in the document.\n    If there is a key called 'FontName' and another key in the same dictionary object\n    that is called 'FontFilex' (where x is null, 2, or 3), then that fontname is\n    embedded.\n\n    We create and add to two sets, fnt = fonts used and emb = fonts embedded.\n    "
    fontkeys = ('/FontFile', '/FontFile2', '/FontFile3')

    def process_font(f: DictionaryObject) -> None:
        if False:
            while True:
                i = 10
        nonlocal fnt, emb
        f = cast(DictionaryObject, f.get_object())
        if '/BaseFont' in f:
            fnt.add(cast(str, f['/BaseFont']))
        if '/CharProcs' in f or ('/FontDescriptor' in f and any((x in cast(DictionaryObject, f['/FontDescriptor']) for x in fontkeys))) or ('/DescendantFonts' in f and '/FontDescriptor' in cast(DictionaryObject, cast(ArrayObject, f['/DescendantFonts'])[0].get_object()) and any((x in cast(DictionaryObject, cast(DictionaryObject, cast(ArrayObject, f['/DescendantFonts'])[0].get_object())['/FontDescriptor']) for x in fontkeys))):
            emb.add(cast(str, f['/BaseFont']))
    if '/DR' in obj and '/Font' in cast(DictionaryObject, obj['/DR']):
        for f in cast(DictionaryObject, cast(DictionaryObject, obj['/DR'])['/Font']):
            process_font(f)
    if '/Resources' in obj:
        if '/Font' in cast(DictionaryObject, obj['/Resources']):
            for f in cast(DictionaryObject, cast(DictionaryObject, obj['/Resources'])['/Font']).values():
                process_font(f)
        if '/XObject' in cast(DictionaryObject, obj['/Resources']):
            for x in cast(DictionaryObject, cast(DictionaryObject, obj['/Resources'])['/XObject']).values():
                _get_fonts_walk(cast(DictionaryObject, x.get_object()), fnt, emb)
    if '/Annots' in obj:
        for a in cast(ArrayObject, obj['/Annots']):
            _get_fonts_walk(cast(DictionaryObject, a.get_object()), fnt, emb)
    if '/AP' in obj:
        if cast(DictionaryObject, cast(DictionaryObject, obj['/AP'])['/N']).get('/Type') == '/XObject':
            _get_fonts_walk(cast(DictionaryObject, cast(DictionaryObject, obj['/AP'])['/N']), fnt, emb)
        else:
            for a in cast(DictionaryObject, cast(DictionaryObject, obj['/AP'])['/N']):
                _get_fonts_walk(cast(DictionaryObject, a), fnt, emb)
    return (fnt, emb)

class _VirtualListImages(Sequence[ImageFile]):

    def __init__(self, ids_function: Callable[[], List[Union[str, List[str]]]], get_function: Callable[[Union[str, List[str], Tuple[str]]], ImageFile]) -> None:
        if False:
            i = 10
            return i + 15
        self.ids_function = ids_function
        self.get_function = get_function
        self.current = -1

    def __len__(self) -> int:
        if False:
            return 10
        return len(self.ids_function())

    def keys(self) -> List[Union[str, List[str]]]:
        if False:
            while True:
                i = 10
        return self.ids_function()

    def items(self) -> List[Tuple[Union[str, List[str]], ImageFile]]:
        if False:
            for i in range(10):
                print('nop')
        return [(x, self[x]) for x in self.ids_function()]

    @overload
    def __getitem__(self, index: Union[int, str, List[str]]) -> ImageFile:
        if False:
            while True:
                i = 10
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ImageFile]:
        if False:
            print('Hello World!')
        ...

    def __getitem__(self, index: Union[int, slice, str, List[str], Tuple[str]]) -> Union[ImageFile, Sequence[ImageFile]]:
        if False:
            print('Hello World!')
        lst = self.ids_function()
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            lst = [lst[x] for x in indices]
            cls = type(self)
            return cls(lambda : lst, self.get_function)
        if isinstance(index, (str, list, tuple)):
            return self.get_function(index)
        if not isinstance(index, int):
            raise TypeError('invalid sequence indices type')
        len_self = len(lst)
        if index < 0:
            index = len_self + index
        if index < 0 or index >= len_self:
            raise IndexError('sequence index out of range')
        return self.get_function(lst[index])

    def __iter__(self) -> Iterator[ImageFile]:
        if False:
            print('Hello World!')
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        p = [f'Image_{i}={n}' for (i, n) in enumerate(self.ids_function())]
        return f"[{', '.join(p)}]"