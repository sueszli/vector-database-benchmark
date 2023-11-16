#
# Copyright 2019 AXA Group Operations S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging


class ParsrOutputInterpreter(object):
    """Functions to interpret Parsr's resultant JSON file, enabling
    access to the underlying document content
    """

    def __init__(self, object=None):
        """Constructor for the class

        - object: the Parsr JSON file to be loaded
        """
        logging.basicConfig(level=logging.DEBUG,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.object = None
        if object is not None:
            self.load_object(object)

    def __get_text_types(self):
        """Internal function returning the types of text structures
        """
        return ['word', 'line', 'character', 'paragraph', 'heading']

    def __text_objects_none_page(self, txts, page_number_none):
        for page in self.object['pages']:
            for element in page['elements']:
                if element['type'] in self.__get_text_types():
                    txts.append(element)

    def __get_text_objects(self, page_number=None):
        texts = []
        if page_number is not None:
            page = self.get_page(page_number)
            if page is None:
                logging.error(
                    "Cannot get text elements for the requested page; Page {} not found".format(page_number))
                return None
            else:
                for element in page['elements']:
                    if element['type'] in self.__get_text_types():
                        texts.append(element)
        else:
            texts = self.__text_object_none_page(texts, page_number)

        return texts

    def __text_from_text_object(self, text_object: dict) -> str:
        result = ""
        if (text_object['type'] in ['paragraph', 'heading']) or (
                text_object['type'] in ['line']):
            for i in text_object['content']:
                result += self.__text_from_text_object(i)
        elif text_object['type'] in ['word']:
            if isinstance(text_object['content'], list):
                for i in text_object['content']:
                    result += self.__text_from_text_object(i)
            else:
                result += text_object['content']
                result += ' '
        elif text_object['type'] in ['character']:
            result += text_object['content']
        return result

    def load_object(self, object):
        self.object = object

    def get_page(self, page_number: int):
        """Get a particular page in a document

        - page_number: The number of the page to be searched
        """
        for p in self.object['pages']:
            if p['pageNumber'] == page_number:
                return p
        logging.error("Page {} not found".format(page_number))
        return None

    def get_text(self, page_number: int = None) -> str:
        """Get the entire text from a particular page

        - page_number: The page number from which all the text is to be
        extracted
        """
        final_text = ""
        for text_obj in self.__get_text_objects(page_number):
            final_text += self.__text_from_text_object(text_obj)
            final_text += "\n\n"
        return final_text
