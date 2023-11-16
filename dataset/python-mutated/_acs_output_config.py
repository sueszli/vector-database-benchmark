class ACSOutputConfig:
    """Config class for creating an Azure Cognitive Services index.

    :param acs_index_name:
    :type acs_index_name: str
    :param acs_connection_id:
    :type acs_connection_id: str
    :param acs_index_content_key:
    :type acs_index_content_key: str
    :param acs_embedding_key:
    :type acs_embedding_key: str
    :param acs_title_key:
    :type acs_title_key: str
    """

    def __init__(self, *, acs_index_name: str=None, acs_connection_id: str=None):
        if False:
            for i in range(10):
                print('nop')
        self.acs_index_name = acs_index_name
        self.acs_connection_id = acs_connection_id