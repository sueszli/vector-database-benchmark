from enum import Enum

class DefaultCategories(Enum):
    HOUSING = 0
    FOOD = 1
    GAS = 2
    SHOPPING = 3
seller_category_map = {}
seller_category_map['Exxon'] = DefaultCategories.GAS
seller_category_map['Target'] = DefaultCategories.SHOPPING

class Categorizer(object):

    def __init__(self, seller_category_map, seller_category_overrides_map):
        if False:
            while True:
                i = 10
        self.seller_category_map = seller_category_map
        self.seller_category_overrides_map = seller_category_overrides_map

    def categorize(self, transaction):
        if False:
            return 10
        if transaction.seller in self.seller_category_map:
            return self.seller_category_map[transaction.seller]
        if transaction.seller in self.seller_category_overrides_map:
            seller_category_map[transaction.seller] = self.manual_overrides[transaction.seller].peek_min()
            return self.seller_category_map[transaction.seller]
        return None

class Transaction(object):

    def __init__(self, timestamp, seller, amount):
        if False:
            return 10
        self.timestamp = timestamp
        self.seller = seller
        self.amount = amount

class Budget(object):

    def __init__(self, template_categories_to_budget_map):
        if False:
            i = 10
            return i + 15
        self.categories_to_budget_map = template_categories_to_budget_map

    def override_category_budget(self, category, amount):
        if False:
            i = 10
            return i + 15
        self.categories_to_budget_map[category] = amount