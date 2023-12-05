# Benchmarking Vector databases on Code embeddings

_topics: information retrieval, databases, measurement & performance analysis_

---

since the 70ies relational databases have proven themselves very useful for information retrieval on sparse representation vectors using inverted indexes.

but because of the recent discovery of learning encoders called “transformers” and their dense representation vectors called "embeddings", we require more sophisticated data structures to be able to search our encodings – such as hierarchical navigable small-world networks (HNSW).

and while many modern "vector databases" already deliver performant crud operations and convenient wrappers for hnsw-operations, traditional databases like postgres or established search engines like apache lucene (as of dec. 2023) yet have to catch up.



## how does it work

the goal of the retrieval problem is to search the top-k passage embeddings with the largest dot products for a given query embedding.
