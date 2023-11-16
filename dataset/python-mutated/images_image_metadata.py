from msrest.serialization import Model

class ImagesImageMetadata(Model):
    """Defines a count of the number of websites where you can shop or perform
    other actions related to the image.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar shopping_sources_count: The number of websites that sell the
     products seen in the image.
    :vartype shopping_sources_count: int
    :ivar recipe_sources_count: The number of websites that offer recipes of
     the food seen in the image.
    :vartype recipe_sources_count: int
    :ivar aggregate_offer: A summary of the online offers of products found in
     the image. For example, if the image is of a dress, the offer might
     identify the lowest price and the number of offers found. Only visually
     similar products insights include this field. The offer includes the
     following fields: Name, AggregateRating, OfferCount, and LowPrice.
    :vartype aggregate_offer:
     ~azure.cognitiveservices.search.visualsearch.models.AggregateOffer
    """
    _validation = {'shopping_sources_count': {'readonly': True}, 'recipe_sources_count': {'readonly': True}, 'aggregate_offer': {'readonly': True}}
    _attribute_map = {'shopping_sources_count': {'key': 'shoppingSourcesCount', 'type': 'int'}, 'recipe_sources_count': {'key': 'recipeSourcesCount', 'type': 'int'}, 'aggregate_offer': {'key': 'aggregateOffer', 'type': 'AggregateOffer'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(ImagesImageMetadata, self).__init__(**kwargs)
        self.shopping_sources_count = None
        self.recipe_sources_count = None
        self.aggregate_offer = None