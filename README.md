# Benchmarking Vector databases on Code embeddings

Topics: information retrieval, databases, measurement & performance analysis.

Course: CS 854, Performance Engineering, Fall 2023 – Ali Mashtizadeh

Authors:

- Raymond Chang, rkchang@uwaterloo.ca, https://github.com/rkchang
- Vikram N. Subramanian, vnsubram@uwaterloo.ca, https://github.com/vikramsubramanian
- Yahya Jabary, yahya.jabary@uwaterloo.ca, https://github.com/sueszli

---

Through simple benchmarks we created an initial support to decide between the many new vector database products on the market, specifically for the purpose of searching Code-embeddings.

To achieve this, we have created a test set that has a realistic dimensioning and size (to our knowledge, we are the very first to have done so). This has enabled us to simulate a realistic workload.


## Why another database?

Since the 1970ies relational databases have been very useful for information retrieval on sparse representation vectors using inverted indexes.

But because of the recent discovery of learning encoders called “transformers” and their dense representation vectors called "embeddings", we require more sophisticated data structures to be able to search our encodings – such as Hierarchical Navigable Small-World Networks (HNSW). HNSW are important, because they allow us to find vectors that are close and "semantically similar" to our query vector.

While many modern "vector databases" already deliver performant CRUD operations and HNSW-search, traditional databases like Postgres or established search engines like Apache Lucene (as of December 2023) yet have to catch up both in terms of performance and convenience [^1].


## Step 1) Generating a corpus of code documents

Our goal was to scrape at least half a million code files off the internet.

The first website that came to our mind was GitHub.

The first method to scrape GitHub involves using their API. However, the issue with this approach is the aggressive rate limiting imposed by GitHub. The rate limits are as follows:

- Anonymous client rate limit: 60 requests per hour
- Authenticated client rate limit: 5000 requests per hour
- Enterprise client rate limit: 15000 requests per hour

There are several clever ways to bypass this issue, such as bandwidth throttling, rotating IP addresses, switching client IDs, spoofing your location, and figuring out how their rate limiter works through trial and error. However, even with the fastest scraper, it would take days to get all the data needed due to the sophistication of their rate limit.

The second method involves web scraping GitHub. Setting a language as a search filter doesn't work because the results are limited to 5 pages. Although you could use random entries to scrape these 5 pages on multiple iterations, it would still be very time-consuming and require lots of manual work.

Two alternative paths were found:

1. Using the `system size:>100000` query to filter out by the largest repositories to get a huge number of files at once. This was done manually, and around 300 links were collected before giving up.
2. Scraping the topic pages for C/C++ to get access to thousands of repositories within the same page, simply by clicking the "load more..." button. This way, you don't even get rate-limited and can run multiple requests that build on top of each other. Around 2000 repository links were scraped this way.

After collecting a bunch of links to repositories, the files were merged and sorted: 2340 original lines, 186 removed, 2154 remaining. The whole process was repeated again, but for Python code which led to 1154 original lines.

Next, all repositories from the collected links were cloned and a script was run to extract all the files from the repositories and put them into a single folder. Another script was run to remove all the files that weren't C/C++ or Python files.

Finally, the files were uploaded by splitting them into 5000-10000 files each, because GitHub has a size limit per commit (otherwise you have to pay for GitHub LFS). This process allowed for the successful scraping of half a million files of code.


## Step 2) Mutating documents to create clusters

Next we wanted to generate clusters of data through a process called mutation. The goal is to create **semantically similar code blocks** that can be grouped together. The mutation process involves inserting blocks of dead code into each function. We generate five mutated versions for each function, with each successive mutation increasing in complexity.

This can be achieved by mutating each function and inserting some dead code, thus preserving the semantics. For each function, we generate five mutated versions such that each successive mutant is more complex than the previous. Mutations have to be similar but not too similar to ensure recall drops. 

Here's an example of a mutation:

```python
# Original
def foo():
    ...

# Mutated
def foo():
    if False:
        i = 0
    ...
```

We were initially unsure about how complicated these mutations would have to be, so we built a fairly general solution. We start by parsing each function and generating an Abstract Syntax Tree (AST). We then traverse the AST and upon finding a function, insert some dead code right at the beginning. We then dump these mutations and their originals into JSON files to be parsed later.

One of the measurements we generated was how recall was affected by load on the database. We wanted to make sure that recall would actually decrease, so we had to make sure the mutations actually affected the **(cosine) similarity**. We did this by converting some mutations into embeddings and measuring them. 

We first had to determine a baseline, which was the largest difference in similarity we would encounter. We did this by comparing the similarity of the two largest functions, which are fairly likely to be the most different semantically, and got a **cosine similarity of 0.68**. We then checked the similarity of the smallest function with its mutations as well as the largest function and found that their similarity does actually differ. The smallest function had a similarity range of **0.82 to 0.92** with its mutations, while the largest function had a range of **0.85 to 0.91**. 

This analysis confirms that our mutation process is effective in generating semantically similar but distinct functions, which is crucial for our goal of data clustering.


## Step 3) Encoding corpus to generate embeddings


## Step 4) Retrieving embeddings through HNSW

The goal of the retrieval problem is to search the top-k passage embeddings with the largest dot products for a given query embedding.

[^1]: https://doi.org/10.48550/arXiv.2308.14963
