"""
*TL;DR
Separates data in GUIs from the ways it is presented, and accepted.
"""
from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def __iter__(self):
        if False:
            return 10
        pass

    @abstractmethod
    def get(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Returns an object with a .items() call method\n        that iterates over key,value pairs of its information.'
        pass

    @property
    @abstractmethod
    def item_type(self):
        if False:
            return 10
        pass

class ProductModel(Model):

    class Price(float):
        """A polymorphic way to pass a float with a particular
        __str__ functionality."""

        def __str__(self):
            if False:
                print('Hello World!')
            return f'{self:.2f}'
    products = {'milk': {'price': Price(1.5), 'quantity': 10}, 'eggs': {'price': Price(0.2), 'quantity': 100}, 'cheese': {'price': Price(2.0), 'quantity': 10}}
    item_type = 'product'

    def __iter__(self):
        if False:
            return 10
        yield from self.products

    def get(self, product):
        if False:
            while True:
                i = 10
        try:
            return self.products[product]
        except KeyError as e:
            raise KeyError(str(e) + " not in the model's item list.")

class View(ABC):

    @abstractmethod
    def show_item_list(self, item_type, item_list):
        if False:
            return 10
        pass

    @abstractmethod
    def show_item_information(self, item_type, item_name, item_info):
        if False:
            print('Hello World!')
        'Will look for item information by iterating over key,value pairs\n        yielded by item_info.items()'
        pass

    @abstractmethod
    def item_not_found(self, item_type, item_name):
        if False:
            while True:
                i = 10
        pass

class ConsoleView(View):

    def show_item_list(self, item_type, item_list):
        if False:
            return 10
        print(item_type.upper() + ' LIST:')
        for item in item_list:
            print(item)
        print('')

    @staticmethod
    def capitalizer(string):
        if False:
            for i in range(10):
                print('nop')
        return string[0].upper() + string[1:].lower()

    def show_item_information(self, item_type, item_name, item_info):
        if False:
            for i in range(10):
                print('nop')
        print(item_type.upper() + ' INFORMATION:')
        printout = 'Name: %s' % item_name
        for (key, value) in item_info.items():
            printout += ', ' + self.capitalizer(str(key)) + ': ' + str(value)
        printout += '\n'
        print(printout)

    def item_not_found(self, item_type, item_name):
        if False:
            for i in range(10):
                print('nop')
        print(f'That {item_type} "{item_name}" does not exist in the records')

class Controller:

    def __init__(self, model, view):
        if False:
            for i in range(10):
                print('nop')
        self.model = model
        self.view = view

    def show_items(self):
        if False:
            print('Hello World!')
        items = list(self.model)
        item_type = self.model.item_type
        self.view.show_item_list(item_type, items)

    def show_item_information(self, item_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show information about a {item_type} item.\n        :param str item_name: the name of the {item_type} item to show information about\n        '
        try:
            item_info = self.model.get(item_name)
        except Exception:
            item_type = self.model.item_type
            self.view.item_not_found(item_type, item_name)
        else:
            item_type = self.model.item_type
            self.view.show_item_information(item_type, item_name, item_info)

def main():
    if False:
        while True:
            i = 10
    '\n    >>> model = ProductModel()\n    >>> view = ConsoleView()\n    >>> controller = Controller(model, view)\n\n    >>> controller.show_items()\n    PRODUCT LIST:\n    milk\n    eggs\n    cheese\n    <BLANKLINE>\n\n    >>> controller.show_item_information("cheese")\n    PRODUCT INFORMATION:\n    Name: cheese, Price: 2.00, Quantity: 10\n    <BLANKLINE>\n\n    >>> controller.show_item_information("eggs")\n    PRODUCT INFORMATION:\n    Name: eggs, Price: 0.20, Quantity: 100\n    <BLANKLINE>\n\n    >>> controller.show_item_information("milk")\n    PRODUCT INFORMATION:\n    Name: milk, Price: 1.50, Quantity: 10\n    <BLANKLINE>\n\n    >>> controller.show_item_information("arepas")\n    That product "arepas" does not exist in the records\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()