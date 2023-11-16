import logging

class ParsrOutputInterpreter(object):
    """Functions to interpret Parsr's resultant JSON file, enabling
    access to the underlying document content
    """

    def __init__(self, object=None):
        if False:
            print('Hello World!')
        'Constructor for the class\n\n        - object: the Parsr JSON file to be loaded\n        '
        logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
        self.object = None
        if object is not None:
            self.load_object(object)

    def __get_text_types(self):
        if False:
            while True:
                i = 10
        'Internal function returning the types of text structures\n        '
        return ['word', 'line', 'character', 'paragraph', 'heading']

    def __text_objects_none_page(self, txts, page_number_none):
        if False:
            print('Hello World!')
        for page in self.object['pages']:
            for element in page['elements']:
                if element['type'] in self.__get_text_types():
                    txts.append(element)

    def __get_text_objects(self, page_number=None):
        if False:
            while True:
                i = 10
        texts = []
        if page_number is not None:
            page = self.get_page(page_number)
            if page is None:
                logging.error('Cannot get text elements for the requested page; Page {} not found'.format(page_number))
                return None
            else:
                for element in page['elements']:
                    if element['type'] in self.__get_text_types():
                        texts.append(element)
        else:
            texts = self.__text_object_none_page(texts, page_number)
        return texts

    def __text_from_text_object(self, text_object: dict) -> str:
        if False:
            return 10
        result = ''
        if text_object['type'] in ['paragraph', 'heading'] or text_object['type'] in ['line']:
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
        if False:
            print('Hello World!')
        self.object = object

    def get_page(self, page_number: int):
        if False:
            for i in range(10):
                print('nop')
        'Get a particular page in a document\n\n        - page_number: The number of the page to be searched\n        '
        for p in self.object['pages']:
            if p['pageNumber'] == page_number:
                return p
        logging.error('Page {} not found'.format(page_number))
        return None

    def get_text(self, page_number: int=None) -> str:
        if False:
            print('Hello World!')
        'Get the entire text from a particular page\n\n        - page_number: The page number from which all the text is to be\n        extracted\n        '
        final_text = ''
        for text_obj in self.__get_text_objects(page_number):
            final_text += self.__text_from_text_object(text_obj)
            final_text += '\n\n'
        return final_text