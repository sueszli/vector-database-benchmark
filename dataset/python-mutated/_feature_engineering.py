from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.toolkits._internal_utils import _toolkit_repr_print, _precomputed_field, _raise_error_if_not_sframe

class TransformerBase(object):
    """
    An abstract base class for user defined Transformers.

    **Overview**

    A Transformer is a stateful object that transforms input data (as an
    SFrame) from one form to another. Transformers are commonly used for
    feature engineering. In addition to the modules provided in Turi
    Create, users can extend the following class and write transformers that
    integrate seamlessly with the already existing ones.

    **Defining Custom Transformers**

    Each transformer object is one must have the following methods:

        +---------------+---------------------------------------------------+
        |   Method      | Description                                       |
        +===============+===================================================+
        | __init__      | Construct the object.                             |
        +---------------+---------------------------------------------------+
        | fit           | Fit the object using training data.               |
        +---------------+---------------------------------------------------+
        | transform     | Transform the object on training/test data.       |
        +---------------+---------------------------------------------------+

    In addition to these methods, there are convenience methods with default
    implementations:

        +---------------+---------------------------------------------------+
        |   Method      | Description                                       |
        +===============+===================================================+
        | fit_transform | First perform fit() and then transform() on data. |
        +---------------+---------------------------------------------------+

    See Also
    --------
    :class:`turicreate.toolkits.feature_engineering.TransformerChain`,
    :func:`turicreate.toolkits.feature_engineering.create`

    Notes
    ------
    - User defined Transformers behave identically to those that are already
      provided. They can be saved/loaded both locally and remotely, can
      be chained together, and can be deployed as components of predictive
      services.

    Examples
    --------

    In this example, we will write a simple Transformer that will subtract
    (for each column) the mean value observed during the `fit` stage.

    .. sourcecode:: python

        import turicreate
        from . import TransformerBase

        class MyTransformer(TransformerBase):

            def __init__(self):
                pass

            def fit(self, dataset):
                ''' Learn means during the fit stage.'''
                self.mean = {}
                for col in dataset.column_names():
                    self.mean[col] = dataset[col].mean()
                return self

            def transform(self, dataset):
                ''' Subtract means during the transform stage.'''
                new_dataset = turicreate.SFrame()
                for col in dataset.column_names():
                    new_dataset[col] = dataset[col] - self.mean[col]
                return new_dataset

        # Create the model
        model = tc.feature_engineering.create(dataset, MyTransformer())

        # Transform new data
        transformed_sf = model.transform(sf)

        # Save and load this model.
        model.save('foo-bar')
        loaded_model = turicreate.load_model('foo-bar')

    """

    def __init__(self, **kwargs):
        if False:
            return 10
        pass

    def _get_summary_struct(self):
        if False:
            i = 10
            return i + 15
        model_fields = []
        for attr in self.__dict__:
            if not attr.startswith('_'):
                model_fields.append((attr, _precomputed_field(getattr(self, attr))))
        return ([model_fields], ['Attributes'])

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        (sections, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, sections, section_titles, width=20)

    def fit(self, data):
        if False:
            for i in range(10):
                print('nop')
        "\n        Fits a transformer using the SFrame `data`.\n\n        Parameters\n        ----------\n        data : SFrame\n            The data used to fit the transformer.\n\n        Returns\n        -------\n        self (A fitted object)\n\n        See Also\n        --------\n        transform, fit_transform\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            my_tr = MyTransformer(features = ['salary', 'age'])\n            my_tr = mt_tr.fit(sf)\n        "
        pass

    def transform(self, data):
        if False:
            return 10
        '\n        Transform the SFrame `data` using a fitted model.\n\n        Parameters\n        ----------\n        data : SFrame\n            The data  to be transformed.\n\n        Returns\n        -------\n        A transformed SFrame.\n\n        Returns\n        -------\n        out: SFrame\n            A transformed SFrame.\n\n        See Also\n        --------\n        fit, fit_transform\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            my_tr = turicreate.feature_engineering.create(train_data,\n                                                        MyTransformer())\n            transformed_sf = my_tr.transform(sf)\n        '
        raise NotImplementedError

    def fit_transform(self, data):
        if False:
            while True:
                i = 10
        '\n        First fit a transformer using the SFrame `data` and then return a transformed\n        version of `data`.\n\n        Parameters\n        ----------\n        data : SFrame\n            The data used to fit the transformer. The same data is then also\n            transformed.\n\n        Returns\n        -------\n        Transformed SFrame.\n\n        See Also\n        --------\n        transform, fit_transform\n\n        Notes\n        -----\n        The default implementation calls `fit` and then calls `transform`.\n        You may override this function with a more efficient implementation."\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            my_tr = MyTransformer()\n            transformed_sf = my_tr.fit_transform(sf)\n        '
        self.fit(data)
        return self.transform(data)

    def _get_instance_and_data(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class Transformer(TransformerBase):
    _fit_examples_doc = '\n    '
    _fit_transform_examples_doc = '\n    '
    _transform_examples_doc = '\n    '

    def __init__(self, model_proxy=None, _class=None):
        if False:
            return 10
        self.__proxy__ = model_proxy
        if _class:
            self.__class__ = _class

    def fit(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Fit a transformer using the SFrame `data`.\n\n        Parameters\n        ----------\n        data : SFrame\n            The data used to fit the transformer.\n\n        Returns\n        -------\n        self (A fitted version of the object)\n\n        See Also\n        --------\n        transform, fit_transform\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n        {examples}\n        '
        _raise_error_if_not_sframe(data, 'data')
        self.__proxy__.fit(data)
        return self

    def transform(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Transform the SFrame `data` using a fitted model.\n\n        Parameters\n        ----------\n        data : SFrame\n            The data  to be transformed.\n\n        Returns\n        -------\n        A transformed SFrame.\n\n        See Also\n        --------\n        fit, fit_transform\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n        {examples}\n\n        '
        _raise_error_if_not_sframe(data, 'data')
        return self.__proxy__.transform(data)

    def fit_transform(self, data):
        if False:
            print('Hello World!')
        '\n        First fit a transformer using the SFrame `data` and then return a\n        transformed version of `data`.\n\n        Parameters\n        ----------\n        data : SFrame\n            The data used to fit the transformer. The same data is then also\n            transformed.\n\n        Returns\n        -------\n        Transformed SFrame.\n\n        See Also\n        --------\n        fit, transform\n\n        Notes\n        ------\n        - Fit transform modifies self.\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n        {examples}\n        '
        _raise_error_if_not_sframe(data, 'data')
        return self.__proxy__.fit_transform(data)

    def _list_fields(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        List of fields stored in the model. Each of these fields can be queried\n        using the ``get(field)`` function or ``m[field]``.\n\n        Returns\n        -------\n        out : list[str]\n            A list of fields that can be queried using the ``get`` method.\n\n        See Also\n        ---------\n        get\n        '
        return self.__proxy__.list_fields()

    def _get(self, field):
        if False:
            for i in range(10):
                print('nop')
        "Return the value for the queried field.\n\n        Each of these fields can be queried in one of two ways:\n\n        >>> out = m['field']\n        >>> out = m.get('field')  # equivalent to previous line\n\n        Parameters\n        ----------\n        field : string\n            Name of the field to be retrieved.\n\n        Returns\n        -------\n        out : value\n            The current value of the requested field.\n        "
        if field in self._list_fields():
            return self.__proxy__.get(field)
        else:
            raise KeyError('Field "%s" not in model. Available fields are %s.' % (field, ', '.join(self._list_fields())))

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.get(key)

    @classmethod
    def _is_gl_pickle_safe(cls):
        if False:
            i = 10
            return i + 15
        '\n        Return True if the model is GLPickle safe i.e if the model does not\n        contain elements that are written using Python + Turi objects.\n        '
        return False

class _SampleTransformer(Transformer):

    def __init__(self, features=None, constant=0.5):
        if False:
            print('Hello World!')
        opts = {}
        opts['features'] = features
        opts['constant'] = constant
        proxy = _tc.extensions._SampleTransformer()
        proxy.init_transformer(opts)
        super(_SampleTransformer, self).__init__(proxy, self.__class__)

    def _get_summary_struct(self):
        if False:
            return 10
        "\n        Returns a structured description of the model, including (where\n        relevant) the schema of the training data, description of the training\n        data, training statistics, and model hyperparameters.\n\n        Returns\n        -------\n        sections : list (of list of tuples)\n            A list of summary sections.\n              Each section is a list.\n                Each item in a section list is a tuple of the form:\n                  ('<label>','<field>')\n\n        section_titles: list\n            A list of section titles.\n              The order matches that of the 'sections' object.\n        "
        section = []
        section_titles = ['Attributes']
        for f in self._list_fields():
            section.append(('%s' % f, '%s' % f))
        return ([section], section_titles)

    def __repr__(self):
        if False:
            return 10
        (section, section_titles) = self._get_summary_struct()
        return _toolkit_repr_print(self, section, section_titles, width=30)