from io import BytesIO

import pytest

from PIL import (
    BmpImagePlugin,
    EpsImagePlugin,
    Image,
    ImageFile,
    UnidentifiedImageError,
    _binary,
    features,
)

from .helper import (
    assert_image,
    assert_image_equal,
    assert_image_similar,
    fromstring,
    hopper,
    skip_unless_feature,
    tostring,
)

# save original block sizes
MAXBLOCK = ImageFile.MAXBLOCK
SAFEBLOCK = ImageFile.SAFEBLOCK


class TestImageFile:
    def test_parser(self):
        def roundtrip(format):
            im = hopper("L").resize((1000, 1000), Image.Resampling.NEAREST)
            if format in ("MSP", "XBM"):
                im = im.convert("1")

            test_file = BytesIO()

            im.copy().save(test_file, format)

            data = test_file.getvalue()

            parser = ImageFile.Parser()
            parser.feed(data)
            im_out = parser.close()

            return im, im_out

        assert_image_equal(*roundtrip("BMP"))
        im1, im2 = roundtrip("GIF")
        assert_image_similar(im1.convert("P"), im2, 1)
        assert_image_equal(*roundtrip("IM"))
        assert_image_equal(*roundtrip("MSP"))
        if features.check("zlib"):
            try:
                # force multiple blocks in PNG driver
                ImageFile.MAXBLOCK = 8192
                assert_image_equal(*roundtrip("PNG"))
            finally:
                ImageFile.MAXBLOCK = MAXBLOCK
        assert_image_equal(*roundtrip("PPM"))
        assert_image_equal(*roundtrip("TIFF"))
        assert_image_equal(*roundtrip("XBM"))
        assert_image_equal(*roundtrip("TGA"))
        assert_image_equal(*roundtrip("PCX"))

        if EpsImagePlugin.has_ghostscript():
            im1, im2 = roundtrip("EPS")
            # This test fails on Ubuntu 12.04, PPC (Bigendian) It
            # appears to be a ghostscript 9.05 bug, since the
            # ghostscript rendering is wonky and the file is identical
            # to that written on ubuntu 12.04 x64
            # md5sum: ba974835ff2d6f3f2fd0053a23521d4a

            # EPS comes back in RGB:
            assert_image_similar(im1, im2.convert("L"), 20)

        if features.check("jpg"):
            im1, im2 = roundtrip("JPEG")  # lossy compression
            assert_image(im1, im2.mode, im2.size)

        with pytest.raises(OSError):
            roundtrip("PDF")

    def test_ico(self):
        with open("Tests/images/python.ico", "rb") as f:
            data = f.read()
        with ImageFile.Parser() as p:
            p.feed(data)
            assert (48, 48) == p.image.size

    @skip_unless_feature("webp")
    @skip_unless_feature("webp_anim")
    def test_incremental_webp(self):
        with ImageFile.Parser() as p:
            with open("Tests/images/hopper.webp", "rb") as f:
                p.feed(f.read(1024))

                # Check that insufficient data was given in the first feed
                assert not p.image

                p.feed(f.read())
            assert (128, 128) == p.image.size

    @skip_unless_feature("zlib")
    def test_safeblock(self):
        im1 = hopper()

        try:
            ImageFile.SAFEBLOCK = 1
            im2 = fromstring(tostring(im1, "PNG"))
        finally:
            ImageFile.SAFEBLOCK = SAFEBLOCK

        assert_image_equal(im1, im2)

    def test_raise_oserror(self):
        with pytest.raises(OSError):
            ImageFile.raise_oserror(1)

    def test_raise_typeerror(self):
        with pytest.raises(TypeError):
            parser = ImageFile.Parser()
            parser.feed(1)

    def test_negative_stride(self):
        with open("Tests/images/raw_negative_stride.bin", "rb") as f:
            input = f.read()
        p = ImageFile.Parser()
        p.feed(input)
        with pytest.raises(OSError):
            p.close()

    def test_no_format(self):
        buf = BytesIO(b"\x00" * 255)

        class DummyImageFile(ImageFile.ImageFile):
            def _open(self):
                self._mode = "RGB"
                self._size = (1, 1)

        im = DummyImageFile(buf)
        assert im.format is None
        assert im.get_format_mimetype() is None

    def test_oserror(self):
        im = Image.new("RGB", (1, 1))
        with pytest.raises(OSError):
            im.save(BytesIO(), "JPEG2000", num_resolutions=2)

    def test_truncated(self):
        b = BytesIO(
            b"BM000000000000"  # head_data
            + _binary.o32le(
                ImageFile.SAFEBLOCK + 1 + 4
            )  # header_size, so BmpImagePlugin will try to read SAFEBLOCK + 1 bytes
            + (
                b"0" * ImageFile.SAFEBLOCK
            )  # only SAFEBLOCK bytes, so that the header is truncated
        )
        with pytest.raises(OSError) as e:
            BmpImagePlugin.BmpImageFile(b)
        assert str(e.value) == "Truncated File Read"

    @skip_unless_feature("zlib")
    def test_truncated_with_errors(self):
        with Image.open("Tests/images/truncated_image.png") as im:
            with pytest.raises(OSError):
                im.load()

            # Test that the error is raised if loaded a second time
            with pytest.raises(OSError):
                im.load()

    @skip_unless_feature("zlib")
    def test_truncated_without_errors(self):
        with Image.open("Tests/images/truncated_image.png") as im:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                im.load()
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = False

    @skip_unless_feature("zlib")
    def test_broken_datastream_with_errors(self):
        with Image.open("Tests/images/broken_data_stream.png") as im:
            with pytest.raises(OSError):
                im.load()

    @skip_unless_feature("zlib")
    def test_broken_datastream_without_errors(self):
        with Image.open("Tests/images/broken_data_stream.png") as im:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            try:
                im.load()
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = False


class MockPyDecoder(ImageFile.PyDecoder):
    def decode(self, buffer):
        # eof
        return -1, 0


class MockPyEncoder(ImageFile.PyEncoder):
    def encode(self, buffer):
        return 1, 1, b""

    def cleanup(self):
        self.cleanup_called = True


xoff, yoff, xsize, ysize = 10, 20, 100, 100


class MockImageFile(ImageFile.ImageFile):
    def _open(self):
        self.rawmode = "RGBA"
        self._mode = "RGBA"
        self._size = (200, 200)
        self.tile = [("MOCK", (xoff, yoff, xoff + xsize, yoff + ysize), 32, None)]


class CodecsTest:
    @classmethod
    def setup_class(cls):
        cls.decoder = MockPyDecoder(None)
        cls.encoder = MockPyEncoder(None)

        def decoder_closure(mode, *args):
            cls.decoder.__init__(mode, *args)
            return cls.decoder

        def encoder_closure(mode, *args):
            cls.encoder.__init__(mode, *args)
            return cls.encoder

        Image.register_decoder("MOCK", decoder_closure)
        Image.register_encoder("MOCK", encoder_closure)


class TestPyDecoder(CodecsTest):
    def test_setimage(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)

        im.load()

        assert self.decoder.state.xoff == xoff
        assert self.decoder.state.yoff == yoff
        assert self.decoder.state.xsize == xsize
        assert self.decoder.state.ysize == ysize

        with pytest.raises(ValueError):
            self.decoder.set_as_raw(b"\x00")

    def test_extents_none(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)
        im.tile = [("MOCK", None, 32, None)]

        im.load()

        assert self.decoder.state.xoff == 0
        assert self.decoder.state.yoff == 0
        assert self.decoder.state.xsize == 200
        assert self.decoder.state.ysize == 200

    def test_negsize(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)
        im.tile = [("MOCK", (xoff, yoff, -10, yoff + ysize), 32, None)]

        with pytest.raises(ValueError):
            im.load()

        im.tile = [("MOCK", (xoff, yoff, xoff + xsize, -10), 32, None)]
        with pytest.raises(ValueError):
            im.load()

    def test_oversize(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)
        im.tile = [("MOCK", (xoff, yoff, xoff + xsize + 100, yoff + ysize), 32, None)]

        with pytest.raises(ValueError):
            im.load()

        im.tile = [("MOCK", (xoff, yoff, xoff + xsize, yoff + ysize + 100), 32, None)]
        with pytest.raises(ValueError):
            im.load()

    def test_decode(self):
        decoder = ImageFile.PyDecoder(None)
        with pytest.raises(NotImplementedError):
            decoder.decode(None)


class TestPyEncoder(CodecsTest):
    def test_setimage(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)

        fp = BytesIO()
        ImageFile._save(
            im, fp, [("MOCK", (xoff, yoff, xoff + xsize, yoff + ysize), 0, "RGB")]
        )

        assert self.encoder.state.xoff == xoff
        assert self.encoder.state.yoff == yoff
        assert self.encoder.state.xsize == xsize
        assert self.encoder.state.ysize == ysize

    def test_extents_none(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)
        im.tile = [("MOCK", None, 32, None)]

        fp = BytesIO()
        ImageFile._save(im, fp, [("MOCK", None, 0, "RGB")])

        assert self.encoder.state.xoff == 0
        assert self.encoder.state.yoff == 0
        assert self.encoder.state.xsize == 200
        assert self.encoder.state.ysize == 200

    def test_negsize(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)

        fp = BytesIO()
        self.encoder.cleanup_called = False
        with pytest.raises(ValueError):
            ImageFile._save(
                im, fp, [("MOCK", (xoff, yoff, -10, yoff + ysize), 0, "RGB")]
            )
        assert self.encoder.cleanup_called

        with pytest.raises(ValueError):
            ImageFile._save(
                im, fp, [("MOCK", (xoff, yoff, xoff + xsize, -10), 0, "RGB")]
            )

    def test_oversize(self):
        buf = BytesIO(b"\x00" * 255)

        im = MockImageFile(buf)

        fp = BytesIO()
        with pytest.raises(ValueError):
            ImageFile._save(
                im,
                fp,
                [("MOCK", (xoff, yoff, xoff + xsize + 100, yoff + ysize), 0, "RGB")],
            )

        with pytest.raises(ValueError):
            ImageFile._save(
                im,
                fp,
                [("MOCK", (xoff, yoff, xoff + xsize, yoff + ysize + 100), 0, "RGB")],
            )

    def test_encode(self):
        encoder = ImageFile.PyEncoder(None)
        with pytest.raises(NotImplementedError):
            encoder.encode(None)

        bytes_consumed, errcode = encoder.encode_to_pyfd()
        assert bytes_consumed == 0
        assert ImageFile.ERRORS[errcode] == "bad configuration"

        encoder._pushes_fd = True
        with pytest.raises(NotImplementedError):
            encoder.encode_to_pyfd()

        with pytest.raises(NotImplementedError):
            encoder.encode_to_file(None, None)

    def test_zero_height(self):
        with pytest.raises(UnidentifiedImageError):
            Image.open("Tests/images/zero_height.j2k")
