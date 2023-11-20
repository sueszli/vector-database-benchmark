from dataclasses import dataclass
import json
from pprint import pprint

from glob import glob
from openai import OpenAI, embeddings

from python.mutater import DESTDIR

client = OpenAI()
MODEL = "text-embedding-ada-002"

def get_embedding(text, model=MODEL):
   text = text.replace("\n", " ")
   response = client.embeddings.create(input = [text], model=model)
   return response.data[0].embedding


@dataclass
class FuncEmbeddingRecord():
   func_name: str
   original_str: str
   mutated_embeddings: list[list[float]] 
   mutated_str: list[str] 


EmbeddingsMap = dict[list[float], list[list[float]]]


'''
Benchmarking for precision:
Two passes
Collect all original embeddings
Send them to DB
Collect all responses and persist
iterate over responses and compare by checking original and comparing with mutated
'''

if __name__ == "__main__":
   # Find files to generate embeddings for
   mutation_files = glob(f"{DESTDIR}/*.json")
   # # Generate embeddings
   embeddings_map: EmbeddingsMap = {}
   for i in range(1):
      file_path = mutation_files[i]
      with open(file_path, "r") as f:
         func_records = json.load(f)
         for rec in func_records:
            orig_embedding = get_embedding(rec["original"])
            embeddings_map[orig_embedding] = []
            for mut in rec["mutated"]:
               mut_embedding = get_embedding(mut)
