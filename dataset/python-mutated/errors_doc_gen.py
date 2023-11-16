import re
from pyspark.errors.error_classes import ERROR_CLASSES_MAP

def generate_errors_doc(output_rst_file_path: str) -> None:
    if False:
        print('Hello World!')
    '\n    Generates a reStructuredText (RST) documentation file for PySpark error classes.\n\n    This function fetches error classes defined in `pyspark.errors.error_classes`\n    and writes them into an RST file. The generated RST file provides an overview\n    of common, named error classes returned by PySpark.\n\n    Parameters\n    ----------\n    output_rst_file_path : str\n        The file path where the RST documentation will be written.\n\n    Notes\n    -----\n    The generated RST file can be rendered using Sphinx to visualize the documentation.\n    '
    header = '..  Licensed to the Apache Software Foundation (ASF) under one\n    or more contributor license agreements.  See the NOTICE file\n    distributed with this work for additional information\n    regarding copyright ownership.  The ASF licenses this file\n    to you under the Apache License, Version 2.0 (the\n    "License"); you may not use this file except in compliance\n    with the License.  You may obtain a copy of the License at\n\n..    http://www.apache.org/licenses/LICENSE-2.0\n\n..  Unless required by applicable law or agreed to in writing,\n    software distributed under the License is distributed on an\n    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n    KIND, either express or implied.  See the License for the\n    specific language governing permissions and limitations\n    under the License.\n\n========================\nError classes in PySpark\n========================\n\nThis is a list of common, named error classes returned by PySpark which are defined at `error_classes.py <https://github.com/apache/spark/blob/master/python/pyspark/errors/error_classes.py>`_.\n\nWhen writing PySpark errors, developers must use an error class from the list. If an appropriate error class is not available, add a new one into the list. For more information, please refer to `Contributing Error and Exception <https://spark.apache.org/docs/latest/api/python/development/contributing.html#contributing-error-and-exception>`_.\n'
    with open(output_rst_file_path, 'w') as f:
        f.write(header + '\n\n')
        for (error_key, error_details) in ERROR_CLASSES_MAP.items():
            f.write(error_key + '\n')
            f.write('-' * len(error_key) + '\n\n')
            messages = error_details['message']
            for message in messages:
                message = re.sub('`(\\()', '`\\\\\\1', message)
                f.write(message + '\n')
            f.write('\n\n')