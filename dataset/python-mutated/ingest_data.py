import logging
from typing import Dict, List, Any, Callable, Optional, Union
import numpy as np
from deeplake.core.dataset import Dataset as DeepLakeDataset
from deeplake.core.vectorstore.vector_search.ingestion.data_ingestion import DataIngestion

def run_data_ingestion(elements: List[Dict[str, Any]], dataset: DeepLakeDataset, ingestion_batch_size: int, num_workers: int, embedding_function: Optional[List[Callable]]=None, embedding_tensor: Optional[List[str]]=None, retry_attempt: int=0, total_samples_processed: int=0, logger: Optional[logging.Logger]=None):
    if False:
        for i in range(10):
            print('nop')
    'Running data ingestion into deeplake dataset.\n\n    Args:\n        elements (List[Dict[str, Any]]): List of dictionaries. Each dictionary contains mapping of\n            names of 4 tensors (i.e. "embedding", "metadata", "ids", "text") to their corresponding values.\n        dataset (DeepLakeDataset): deeplake dataset object.\n        embedding_function (Optional, List[Callable]]): function used to convert query into an embedding.\n        embedding_tensor (Optional, List[str]) : tensor name where embedded data will be stored. Defaults to None.\n        ingestion_batch_size (int): The batch size to use during ingestion.\n        num_workers (int): The number of workers to use for ingesting data in parallel.\n        retry_attempt (int): The number of retry attempts already passed.\n        total_samples_processed (int): The number of samples processed before transforms stopped. Defaults to 0.\n        logger (Optional[logging.Logger]): logger where all warnings are logged. Defaults to None.\n    '
    data_ingestion = DataIngestion(elements=elements, dataset=dataset, embedding_function=embedding_function, embedding_tensor=embedding_tensor, ingestion_batch_size=ingestion_batch_size, num_workers=num_workers, retry_attempt=retry_attempt, total_samples_processed=total_samples_processed, logger=logger)
    data_ingestion.run()