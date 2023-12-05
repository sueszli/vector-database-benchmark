# Benchmarking Vector databases on Code embeddings

_Topics: information retrieval, databases, measurement & performance analysis._

---

Since the 1970ies relational databases have been very useful for information retrieval on sparse representation vectors using inverted indexes.

But because of the recent discovery of learning encoders called “transformers” and their dense representation vectors called "embeddings", we require more sophisticated data structures to be able to search our encodings – such as Hierarchical Navigable Small-World Networks (HNSW).

While many modern "vector databases" already deliver performant CRUD operations and convenient wrappers for HNSW-operations, traditional databases like Postgres or established search engines like Apache Lucene (as of December 2023) yet have to catch up.


## Step 1) Generating a corpus of code documents

## Step 2) Mutating documents to create clusters

## Step 3) Encoding corpus to generate embeddings

## Step 4) Retrieving embeddings through HNSW

The goal of the retrieval problem is to search the top-k passage embeddings with the largest dot products for a given query embedding.
