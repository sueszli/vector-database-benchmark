"""Top-level component that wraps the entire app."""
from reflex.components.component import Component
from .bare import Bare

class AppWrap(Bare):
    """Top-level component that wraps the entire app."""

    @classmethod
    def create(cls) -> Component:
        if False:
            for i in range(10):
                print('nop')
        'Create a new AppWrap component.\n\n        Returns:\n            A new AppWrap component containing {children}.\n        '
        return super().create(contents='{children}')