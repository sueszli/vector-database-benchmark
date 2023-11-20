import ast
import logging
import sys
import multiprocessing as mp
import os
import json
from glob import glob
from copy import deepcopy

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
""",
    """
if False:
    # 1
    i = 10
    # 2
    while(True):
        print("Mutation")
""",
    """
if False:
    # 1
    i = 10
    # 2
    while(True):
        print("Mutation")
    # 3
    n = 10
    # Compute nth fibbonacci number
    dp = [0,1]
    for i in range(2,n+1):
        dp.append(dp[i-1] + dp[i-2])
    print(dp[n])
""",
    """
if False:
    # 1
    i = 10
    # 2
    while(True):
        print("Mutation")
    # 3
    n = 10
    # Compute nth fibbonacci number
    dp = [0,1]
    for i in range(2,n+1):
        dp.append(dp[i-1] + dp[i-2])
    print(dp[n])
    # 4
    def dfs(node):
        if node == None:
            return []
        left = dfs(node.left)
        right = dfs(node.right)
""",
    """
if False:
    # 1
    i = 10
    # 2
    while(True):
        print("Mutation")
    # 3
    # Compute nth fibbonacci number
    dp = [0,1]
    for i in range(2,n+1):
        dp.append(dp[i-1] + dp[i-2])
    print(dp[n])
    # 4
    def dfs(node):
        if node == None:
            return []
        left = dfs(node.left)
        right = dfs(node.right)
    # 5
    length = 15
    if length <= 0:
        return []
    elif length == 1:
        return [0]

    sequence = [0, 1]
    while len(sequence) < length:
        next_value = sequence[-1] + sequence[-2]
        sequence.append(next_value)

    return sequence
""",
]

# ------- Mutation logic -------


class Mutater(ast.NodeTransformer):
    """Mutates visited AST"""

    def __init__(self, file_name: str):
        self.mutations: list[dict] = []
        self.file_name = file_name

    def visit_FunctionDef(self, node):
        """Insert a mutation from MUTATIONS at the beginning of the function"""
        original = ast.unparse(node)
        func_mutations = []
        for mutation in MUTATIONS:
            mutation = ast.parse(mutation).body[0]
            new_node = deepcopy(node)
            new_node.body.insert(0, mutation)
            ast.fix_missing_locations(new_node)
            func_mutations.append(ast.unparse(new_node))
        self.generic_visit(node)
        self.mutations.append(
            {"func_name": node.name, "original": original, "mutated": func_mutations}
        )
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
        mutater = Mutater(file_path)
        tree = mutater.visit(tree)
        if len(mutater.mutations) == 0:
            log.debug(f"Did not mutate file '{file_path}'")
            return False
    except Exception as e:
        # So many things can go wrong here just record the ones that failed
        log.info(f"Did not mutate file '{file_path}': {e}")
        return False
    # Convert mutated AST to string
    source = ast.unparse(tree)
    # Dump mutations to file as JSON
    new_file_path = os.path.join(
        DESTDIR, os.path.basename(file_path).strip(".py") + ".json"
    )
    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(mutater.mutations, f, indent=4)
    return True


def main():
    target_dir = sys.argv[1]
    assert os.path.isdir(target_dir)
    # Find files to mutate in target_dir
    target_files = glob(f"{target_dir}/*.py")
    log.info(f"Found {len(target_files)} files in '{target_dir}' to mutate")
    # Mutate all found files in parallel
    with mp.Pool() as pool:
        mutated = pool.map(mutate_file, target_files)
    log.info(f"Mutated {sum(mutated)}/{len(target_files)}")


if __name__ == "__main__":
    main()
