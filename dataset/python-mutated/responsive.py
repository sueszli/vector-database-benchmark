"""Responsive components."""
from reflex.components.layout.box import Box

def mobile_only(*children, **props):
    if False:
        return 10
    'Create a component that is only visible on mobile.\n\n    Args:\n        *children: The children to pass to the component.\n        **props: The props to pass to the component.\n\n    Returns:\n        The component.\n    '
    return Box.create(*children, **props, display=['block', 'none', 'none', 'none'])

def tablet_only(*children, **props):
    if False:
        while True:
            i = 10
    'Create a component that is only visible on tablet.\n\n    Args:\n        *children: The children to pass to the component.\n        **props: The props to pass to the component.\n\n    Returns:\n        The component.\n    '
    return Box.create(*children, **props, display=['none', 'block', 'block', 'none'])

def desktop_only(*children, **props):
    if False:
        i = 10
        return i + 15
    'Create a component that is only visible on desktop.\n\n    Args:\n        *children: The children to pass to the component.\n        **props: The props to pass to the component.\n\n    Returns:\n        The component.\n    '
    return Box.create(*children, **props, display=['none', 'none', 'none', 'block'])

def tablet_and_desktop(*children, **props):
    if False:
        print('Hello World!')
    'Create a component that is only visible on tablet and desktop.\n\n    Args:\n        *children: The children to pass to the component.\n        **props: The props to pass to the component.\n\n    Returns:\n        The component.\n    '
    return Box.create(*children, **props, display=['none', 'block', 'block', 'block'])

def mobile_and_tablet(*children, **props):
    if False:
        while True:
            i = 10
    'Create a component that is only visible on mobile and tablet.\n\n    Args:\n        *children: The children to pass to the component.\n        **props: The props to pass to the component.\n\n    Returns:\n        The component.\n    '
    return Box.create(*children, **props, display=['block', 'block', 'block', 'none'])