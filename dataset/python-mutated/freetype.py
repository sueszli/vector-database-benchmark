"""Enhanced Pygame module for loading and rendering computer fonts"""
from pygame._freetype import Font, STYLE_NORMAL, STYLE_OBLIQUE, STYLE_STRONG, STYLE_UNDERLINE, STYLE_WIDE, STYLE_DEFAULT, init, quit, get_init, was_init, get_cache_size, get_default_font, get_default_resolution, get_error, get_version, set_default_resolution
from pygame.sysfont import match_font, get_fonts, SysFont as _SysFont
__all__ = ['Font', 'STYLE_NORMAL', 'STYLE_OBLIQUE', 'STYLE_STRONG', 'STYLE_UNDERLINE', 'STYLE_WIDE', 'STYLE_DEFAULT', 'init', 'quit', 'get_init', 'was_init', 'get_cache_size', 'get_default_font', 'get_default_resolution', 'get_error', 'get_version', 'set_default_resolution', 'match_font', 'get_fonts']

def SysFont(name, size, bold=False, italic=False, constructor=None):
    if False:
        while True:
            i = 10
    'pygame.ftfont.SysFont(name, size, bold=False, italic=False, constructor=None) -> Font\n    Create a pygame Font from system font resources.\n\n    This will search the system fonts for the given font\n    name. You can also enable bold or italic styles, and\n    the appropriate system font will be selected if available.\n\n    This will always return a valid Font object, and will\n    fallback on the builtin pygame font if the given font\n    is not found.\n\n    Name can also be an iterable of font names, a string of\n    comma-separated font names, or a bytes of comma-separated\n    font names, in which case the set of names will be searched\n    in order. Pygame uses a small set of common font aliases. If the\n    specific font you ask for is not available, a reasonable\n    alternative may be used.\n\n    If optional constructor is provided, it must be a function with\n    signature constructor(fontpath, size, bold, italic) which returns\n    a Font instance. If None, a pygame.freetype.Font object is created.\n    '
    if constructor is None:

        def constructor(fontpath, size, bold, italic):
            if False:
                return 10
            font = Font(fontpath, size)
            font.strong = bold
            font.oblique = italic
            return font
    return _SysFont(name, size, bold, italic, constructor)