# Benchmarking Vector databases on Code embeddings

_topics: information retrieval, databases, measurement & performance analysis_

---

since the 70ies relational databases have proven themselves very useful for information retrieval on sparse representation vectors using inverted indexes.

but since the recent popularity in learning encoders called “transformers” and their dense representation vectors called "embeddings", we require more sophisticated data structures to be able to search our encodings – such as hierarchical navigable small-world networks (HNSW),

and while HNSW are ideal for these kinds of applications, traditional databases like postgres or established search engines like apache lucene (as of dec. 2023) have yet to prove themselves.

- 

??? why is this significant

??? what did we accomplish

??? what's the gist of it all

## introduction

the goal is to search the top-k passage embeddings with the largest dot products for a given query embedding.





inverted indexes for sparse representation vectors
