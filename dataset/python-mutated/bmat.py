"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.vstack import vstack

def bmat(block_lists):
    if False:
        return 10
    'Constructs a block matrix.\n\n    Takes a list of lists. Each internal list is stacked horizontally.\n    The internal lists are stacked vertically.\n\n    Parameters\n    ----------\n    block_lists : list of lists\n        The blocks of the block matrix.\n\n    Return\n    ------\n    CVXPY expression\n        The CVXPY expression representing the block matrix.\n    '
    row_blocks = [hstack(blocks) for blocks in block_lists]
    return vstack(row_blocks)