from django_filters.constants import EMPTY_VALUES
from django_filters.filters import FilterMethod
from .typed_filter import TypedFilter

class ArrayFilterMethod(FilterMethod):

    def __call__(self, qs, value):
        if False:
            print('Hello World!')
        if value is None:
            return qs
        return self.method(qs, self.f.field_name, value)

class ArrayFilter(TypedFilter):
    """
    Filter made for PostgreSQL ArrayField.
    """

    @TypedFilter.method.setter
    def method(self, value):
        if False:
            return 10
        "\n        Override method setter so that in case a custom `method` is provided\n        (see documentation https://django-filter.readthedocs.io/en/stable/ref/filters.html#method),\n        it doesn't fall back to checking if the value is in `EMPTY_VALUES` (from the `__call__` method\n        of the `FilterMethod` class) and instead use our ArrayFilterMethod that consider empty lists as values.\n\n        Indeed when providing a `method` the `filter` method below is overridden and replaced by `FilterMethod(self)`\n        which means that the validation of the empty value is made by the `FilterMethod.__call__` method instead.\n        "
        TypedFilter.method.fset(self, value)
        if value is not None:
            self.filter = ArrayFilterMethod(self)

    def filter(self, qs, value):
        if False:
            i = 10
            return i + 15
        "\n        Override the default filter class to check first whether the list is\n        empty or not.\n        This needs to be done as in this case we expect to get the filter applied with\n        an empty list since it's a valid value but django_filter consider an empty list\n        to be an empty input value (see `EMPTY_VALUES`) meaning that\n        the filter does not need to be applied (hence returning the original\n        queryset).\n        "
        if value in EMPTY_VALUES and value != []:
            return qs
        if self.distinct:
            qs = qs.distinct()
        lookup = f'{self.field_name}__{self.lookup_expr}'
        qs = self.get_method(qs)(**{lookup: value})
        return qs