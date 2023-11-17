import time

import openai.embeddings_utils
from loguru import logger


def _openai_embedding(code: str, engine: str = "pakshi") -> str:
    """
    Get the embedding for a code snippet using the OpenAI API
    """
    try:
        logger.debug("Calling OpenAI embedding API")
        # time.sleep(0.3)
        return [openai.embeddings_utils.get_embedding(code, engine=engine)]

    except openai.error.InvalidRequestError as e:
        logger.error("InvalidRequestError: " + str(e))
        logger.error("About to be chunked: " + code)
        # Handle the token limit exceeded case by chunking the code into smaller parts
        # and then possibly retrying.
        # Note: Implement `chunk_code` to split the code into smaller parts that comply with the token limit.
        smaller_chunks = [code[0 : len(code) // 2], code[len(code) // 2 :]]
        embeddings = [
            openai.embeddings_utils.get_embedding(chunk, engine=engine)
            for chunk in smaller_chunks
        ]
        return embeddings


def get_embedding(data: list[str], engine: str = "pakshi"):
    ret = []
    for x in data:
        embedding = _openai_embedding(x, engine=engine)
        if embedding:
            for i in embedding:
                ret.append(i)
    return ret

if __name__ == "__main__":
    print(get_embedding(["print('hello world')"]))