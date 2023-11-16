#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This implementation of ReadBaseTransformer is responsible for reading a Dictionary object
"""
import io
import typing

from borb.io.read.transformer import ReadTransformerState
from borb.io.read.transformer import Transformer
from borb.io.read.types import AnyPDFType
from borb.io.read.types import Dictionary
from borb.pdf.canvas.event.event_listener import EventListener


class DictionaryTransformer(Transformer):
    """
    This implementation of ReadBaseTransformer is responsible for reading a Dictionary object
    """

    #
    # CONSTRUCTOR
    #

    #
    # PRIVATE
    #

    #
    # PUBLIC
    #

    def can_be_transformed(
        self,
        object: typing.Union[io.BufferedIOBase, io.RawIOBase, io.BytesIO, AnyPDFType],
    ) -> bool:
        """
        This function returns True if the object to be transformed is a Dictionary object
        """
        return isinstance(object, Dictionary)

    def transform(
        self,
        object_to_transform: typing.Union[io.BufferedIOBase, io.RawIOBase, AnyPDFType],
        parent_object: typing.Any,
        context: typing.Optional[ReadTransformerState] = None,
        event_listeners: typing.List[EventListener] = [],
    ) -> typing.Any:
        """
        This function reads a Dictionary from a byte stream
        """

        # create root object
        # fmt: off
        assert isinstance(object_to_transform, Dictionary), "object_to_transform must be of type Dictionary"
        object_to_transform.set_parent(parent_object)
        # fmt: on

        # transform key/value pair(s)
        for k, v in object_to_transform.items():
            v = self.get_root_transformer().transform(
                v, object_to_transform, context, event_listeners
            )
            if v is not None:
                object_to_transform[k] = v

        # return
        return object_to_transform
