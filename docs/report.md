# Benchmarking Vector databases on Code embeddings

Topics: information retrieval, databases, measurement & performance analysis.

Repository: https://github.com/sueszli/vector-database-benchmark

Authors:

- Raymond Chang, rkchang@uwaterloo.ca
- Vikram N. Subramanian, vnsubram@uwaterloo.ca
- Yahya Jabary, yahya.jabary@uwaterloo.ca

---

Since the 1970ies relational databases have been very useful for information retrieval on sparse representation vectors using inverted indexes.

But because of the recent discovery of learning encoders called “transformers” and their dense representation vectors called "embeddings", we require more sophisticated data structures to be able to search our encodings – such as Hierarchical Navigable Small-World Networks (HNSW). HNSW are important, because they allow us to find vectors that are close and "semantically similar" to our query vector.

While many modern "vector databases" already deliver performant CRUD operations and HNSW-search, traditional databases like Postgres or established search engines like Apache Lucene (as of December 2023) yet have to catch up both in terms of performance and convenience [^1].

Through simple benchmarks we want to create an initial support to decide between these new vector database products, specifically for Code-embeddings.

## Step 1) Generating a corpus of code documents

## Step 2) Mutating documents to create clusters

## Step 3) Encoding corpus to generate embeddings

## Step 4) Retrieving embeddings through HNSW

The goal of the retrieval problem is to search the top-k passage embeddings with the largest dot products for a given query embedding.

[^1]: https://doi.org/10.48550/arXiv.2308.14963
