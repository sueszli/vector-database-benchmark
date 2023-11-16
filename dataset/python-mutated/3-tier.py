"""
*TL;DR
Separates presentation, application processing, and data management functions.
"""
from typing import Dict, KeysView, Optional, Union

class Data:
    """Data Store Class"""
    products = {'milk': {'price': 1.5, 'quantity': 10}, 'eggs': {'price': 0.2, 'quantity': 100}, 'cheese': {'price': 2.0, 'quantity': 10}}

    def __get__(self, obj, klas):
        if False:
            i = 10
            return i + 15
        print('(Fetching from Data Store)')
        return {'products': self.products}

class BusinessLogic:
    """Business logic holding data store instances"""
    data = Data()

    def product_list(self) -> KeysView[str]:
        if False:
            for i in range(10):
                print('nop')
        return self.data['products'].keys()

    def product_information(self, product: str) -> Optional[Dict[str, Union[int, float]]]:
        if False:
            while True:
                i = 10
        return self.data['products'].get(product, None)

class Ui:
    """UI interaction class"""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.business_logic = BusinessLogic()

    def get_product_list(self) -> None:
        if False:
            i = 10
            return i + 15
        print('PRODUCT LIST:')
        for product in self.business_logic.product_list():
            print(product)
        print('')

    def get_product_information(self, product: str) -> None:
        if False:
            return 10
        product_info = self.business_logic.product_information(product)
        if product_info:
            print('PRODUCT INFORMATION:')
            print(f'Name: {product.title()}, ' + f"Price: {product_info.get('price', 0):.2f}, " + f"Quantity: {product_info.get('quantity', 0):}")
        else:
            print(f"That product '{product}' does not exist in the records")

def main():
    if False:
        while True:
            i = 10
    '\n    >>> ui = Ui()\n    >>> ui.get_product_list()\n    PRODUCT LIST:\n    (Fetching from Data Store)\n    milk\n    eggs\n    cheese\n    <BLANKLINE>\n\n    >>> ui.get_product_information("cheese")\n    (Fetching from Data Store)\n    PRODUCT INFORMATION:\n    Name: Cheese, Price: 2.00, Quantity: 10\n\n    >>> ui.get_product_information("eggs")\n    (Fetching from Data Store)\n    PRODUCT INFORMATION:\n    Name: Eggs, Price: 0.20, Quantity: 100\n\n    >>> ui.get_product_information("milk")\n    (Fetching from Data Store)\n    PRODUCT INFORMATION:\n    Name: Milk, Price: 1.50, Quantity: 10\n\n    >>> ui.get_product_information("arepas")\n    (Fetching from Data Store)\n    That product \'arepas\' does not exist in the records\n    '
if __name__ == '__main__':
    import doctest
    doctest.testmod()