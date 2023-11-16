""" Utils module to provide helper for our common json operations.

"""
from __future__ import absolute_import
import json
from .FileOperations import getFileContents, openTextFile

def loadJsonFromFilename(filename):
    if False:
        print('Hello World!')
    try:
        return json.loads(getFileContents(filename))
    except ValueError:
        return None

def writeJsonToFilename(filename, contents):
    if False:
        while True:
            i = 10
    with openTextFile(filename, 'w') as output:
        json.dump(contents, output)