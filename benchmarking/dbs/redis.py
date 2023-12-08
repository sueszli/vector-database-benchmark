import redis
from typing import List

class RedisVDB():

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redis_conn = redis.Redis(host=host, port=port, db=db)

    def add(self, repo_name: str, chunk: Chunk):
        self.redis_conn.hset(repo_name, chunk.id, chunk.data)

    def add_multiple(self, repo_name: str, chunks: List[Chunk]):
        with self.redis_conn.pipeline() as pipe:
            for chunk in chunks:
                pipe.hset(repo_name, chunk.id, chunk.data)
            pipe.execute()


# redis_vdb = RedisVDB()
# chunk1 = Chunk('sample_data_1')
# redis_vdb.add('my_repo', chunk1)

# chunks = [Chunk(f'sample_data_{i}') for i in range(2, 5)]
# redis_vdb.add_multiple('my_repo', chunks)

# query_result = redis_vdb.query('my_repo', [/* some query embedding */])
