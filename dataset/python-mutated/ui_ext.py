import json
from typing import Optional, Union

def _clean(d: dict) -> dict:
    if False:
        print('Hello World!')
    return {k: v for (k, v) in d.items() if v is not None}

def boxes(*args: str) -> str:
    if False:
        return 10
    "Create a specification for card's `box` attribute. Indicates where and how to position a card for various layouts.\n\n    Args:\n        args: Either the name of the zone to place the card in, or a specification created using the `box()` function.\n    Returns:\n        A string intended to be used as a card's `box` attribute.\n    "
    return json.dumps(args)

def box(zone: str, order: Optional[int]=None, size: Optional[Union[str, int]]=None, width: Optional[str]=None, height: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    "Create a specification for card's `box` attribute. Indicates where and how to position a card.\n\n    Args:\n        zone: The name of the zone to place the card in.\n        order: An number that determines the position of this card relative to other cards in the same zone. Cards in the same zone are sorted in ascending `order` and then placed left to right (or top to bottom).\n        size: A number that indicates the ratio of available width or height occupied by this card. Defaults to 1 if size, width and height are not provided.\n        width: The width of this card, e.g. `200px`, `50%`, etc.\n        height: The height of this card, e.g. `200px`, `50%`, etc.\n    Returns:\n        A string intended to be used as a card's `box` attribute.\n    "
    if size is not None:
        if not isinstance(size, (int, str)):
            raise ValueError('size must be str or int')
        if isinstance(size, int):
            size = str(size)
    return json.dumps(_clean(dict(zone=zone, order=order, size=size, width=width, height=height)))