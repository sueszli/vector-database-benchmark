"""The category mismatch train-test check module."""
import warnings
from .new_category_train_test import NewCategoryTrainTest

class CategoryMismatchTrainTest(NewCategoryTrainTest):
    """Find new categories in the test set."""

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        warnings.warn('CategoryMismatchTrainTest is deprecated, use NewCategoryTrainTest instead', DeprecationWarning)
    pass