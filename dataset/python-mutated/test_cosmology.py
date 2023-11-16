import pytest
from astropy.cosmology._io.cosmology import from_cosmology, to_cosmology
from .base import IODirectTestBase, ToFromTestMixinBase

class ToFromCosmologyTestMixin(ToFromTestMixinBase):
    """
    Tests for a Cosmology[To/From]Format with ``format="astropy.cosmology"``.
    This class will not be directly called by :mod:`pytest` since its name does
    not begin with ``Test``. To activate the contained tests this class must
    be inherited in a subclass. Subclasses must define a :func:`pytest.fixture`
    ``cosmo`` that returns/yields an instance of a |Cosmology|.
    See ``TestCosmology`` for an example.
    """

    def test_to_cosmology_default(self, cosmo, to_format):
        if False:
            print('Hello World!')
        'Test cosmology -> cosmology.'
        newcosmo = to_format('astropy.cosmology')
        assert newcosmo is cosmo

    def test_from_not_cosmology(self, cosmo, from_format):
        if False:
            while True:
                i = 10
        'Test incorrect type in ``Cosmology``.'
        with pytest.raises(TypeError):
            from_format('NOT A COSMOLOGY', format='astropy.cosmology')

    def test_from_cosmology_default(self, cosmo, from_format):
        if False:
            print('Hello World!')
        'Test cosmology -> cosmology.'
        newcosmo = from_format(cosmo)
        assert newcosmo is cosmo

    @pytest.mark.parametrize('format', [True, False, None, 'astropy.cosmology'])
    def test_is_equivalent_to_cosmology(self, cosmo, to_format, format):
        if False:
            return 10
        "Test :meth:`astropy.cosmology.Cosmology.is_equivalent`.\n\n        This test checks that Cosmology equivalency can be extended to any\n        Python object that can be converted to a Cosmology -- in this case\n        a Cosmology! Since it's the identity conversion, the cosmology is\n        always equivalent to itself, regardless of ``format``.\n        "
        obj = to_format('astropy.cosmology')
        assert obj is cosmo
        is_equiv = cosmo.is_equivalent(obj, format=format)
        assert is_equiv is True

class TestToFromCosmology(IODirectTestBase, ToFromCosmologyTestMixin):
    """Directly test ``to/from_cosmology``."""

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        self.functions = {'to': to_cosmology, 'from': from_cosmology}