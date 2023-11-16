"""Utilities related to QWebHistory."""
from typing import Any, List, Mapping
from qutebrowser.qt.core import QByteArray, QDataStream, QIODevice, QUrl
from qutebrowser.utils import qtutils

def _serialize_items(items, current_idx, stream):
    if False:
        print('Hello World!')
    data = {'currentItemIndex': current_idx, 'history': []}
    for item in items:
        data['history'].append(_serialize_item(item))
    stream.writeInt(3)
    stream.writeQVariantMap(data)

def _serialize_item(item):
    if False:
        return 10
    data = {'originalURLString': item.original_url.toString(QUrl.ComponentFormattingOption.FullyEncoded), 'scrollPosition': {'x': 0, 'y': 0}, 'title': item.title, 'urlString': item.url.toString(QUrl.ComponentFormattingOption.FullyEncoded)}
    try:
        data['scrollPosition']['x'] = item.user_data['scroll-pos'].x()
        data['scrollPosition']['y'] = item.user_data['scroll-pos'].y()
    except (KeyError, TypeError):
        pass
    return data

def serialize(items):
    if False:
        for i in range(10):
            print('nop')
    "Serialize a list of TabHistoryItems to a data stream.\n\n    Args:\n        items: An iterable of TabHistoryItems.\n\n    Return:\n        A (stream, data, user_data) tuple.\n            stream: The reset QDataStream.\n            data: The QByteArray with the raw data.\n            user_data: A list with each item's user data.\n\n    Warning:\n        If 'data' goes out of scope, reading from 'stream' will result in a\n        segfault!\n    "
    data = QByteArray()
    stream = QDataStream(data, QIODevice.OpenModeFlag.ReadWrite)
    user_data: List[Mapping[str, Any]] = []
    current_idx = None
    for (i, item) in enumerate(items):
        if item.active:
            if current_idx is not None:
                raise ValueError('Multiple active items ({} and {}) found!'.format(current_idx, i))
            current_idx = i
    if items:
        if current_idx is None:
            raise ValueError('No active item found!')
    else:
        current_idx = 0
    _serialize_items(items, current_idx, stream)
    user_data += [item.user_data for item in items]
    stream.device().reset()
    qtutils.check_qdatastream(stream)
    return (stream, data, user_data)