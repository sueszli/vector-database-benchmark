import ast
import logging
import sys
import multiprocessing as mp
import os
import random
from glob import glob
from typing import Optional

"""
Mutates all python files in provided directory and dumps their mutated versions into DESTDIR
"""


logging.basicConfig(
    filename="mutation.log",
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p:",
)
log = logging.getLogger(__name__)

# ------- CONSTANTS -------

# Directory where we should store mutated files
DESTDIR = "dataset/python-mutated"

# Mutations that we should insert
MUTATIONS = [
    """
if False:
    i = 10
    return i + 15
""",
    """
if False:
    print("Hello World!")
""",
    """
if False:
    for i in range(10):
        print("nop")
""",
    """
if False:
    while(True):
        i = 10
""",
    """
if False:
    return 10
""",
]

# ------- Mutation logic -------


class Mutate(ast.NodeTransformer):
    """Mutates visited AST"""

    def __init__(self):
        self.mutated = False

    def visit_FunctionDef(self, node):
        """Insert a mutation from MUTATIONS at the beginning of the function"""
        mutation_idx = random.randint(0, len(MUTATIONS) - 1)
        mutation = ast.parse(MUTATIONS[mutation_idx]).body[0]
        node.body.insert(0, mutation)
        ast.fix_missing_locations(node)
        self.mutated = True
        self.generic_visit(node)
        return node


def mutate_file(file_path: str) -> bool:
    assert file_path.endswith(".py")
    log.debug(f"Mutating file '{file_path}'")
    try:
        # Read source code
        f = open(file_path, "r", encoding="utf-8")
        source = f.read()
        f.close()
        # Parse AST
        tree = ast.parse(source)
        # Mutate AST
        mutater = Mutate()
        tree = mutater.visit(tree)
        if not mutater.mutated:
            log.debug(f"Did not mutate file '{file_path}'")
            return False
    except Exception as e:
        # So many things can go wrong here just record the ones that failed
        log.info(f"Did not mutate file '{file_path}': {e}")
        return False
    # Convert mutated AST to string
    source = ast.unparse(tree)
    # Write mutated source code to file with _mutated.py suffix
    new_file_path = os.path.join(DESTDIR, os.path.basename(file_path))
    with open(new_file_path, "w", encoding="utf-8") as f:
        f.write(source)
    return True


def main():
    target_dir = sys.argv[1]
    assert os.path.isdir(target_dir)
    target_files = glob(f"{target_dir}/*.py")
    log.info(f"Found {len(target_files)} files in '{target_dir}' to mutate")
    # Mutate all found files in parallel
    with mp.Pool() as pool:
        mutated = pool.map(mutate_file, target_files)
    log.info(f"Mutated {sum(mutated)}/{len(target_files)}")


if __name__ == "__main__":
    main()
