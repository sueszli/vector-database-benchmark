from typing import List
from pathlib import Path
from bigdl.llm.libs.chatglm_C import Pipeline, GenerationConfig

class ChatGLMContext:

    def __init__(self, pipeline: Pipeline, config: GenerationConfig):
        if False:
            return 10
        self.pipeline = pipeline
        self.config = config

def chatglm_load(path: str, n_ctx: int, n_threads: int, use_mmap: bool=False) -> ChatGLMContext:
    if False:
        while True:
            i = 10
    path = str(Path(path))
    pipeline = Pipeline(path, use_mmap)
    config = GenerationConfig(max_length=n_ctx, num_threads=n_threads)
    return ChatGLMContext(pipeline, config)

def chatglm_tokenize(ctx: ChatGLMContext, prompt: str) -> List[int]:
    if False:
        print('Hello World!')
    return ctx.pipeline.tokenizer.encode(prompt)

def chatglm_detokenize(ctx: ChatGLMContext, input_ids: List[int]) -> str:
    if False:
        for i in range(10):
            print('nop')
    return ctx.pipeline.tokenizer.decode(input_ids)

def chatglm_forward(ctx: ChatGLMContext, input_ids: List[int], do_sample: bool=True, top_k: int=0, top_p: float=0.7, temperature: float=0.95) -> int:
    if False:
        for i in range(10):
            print('nop')
    ctx.config.do_sample = do_sample
    ctx.config.top_k = top_k
    ctx.config.top_p = top_p
    ctx.config.temperature = temperature
    return ctx.pipeline.forward(input_ids, ctx.config)

def chatglm_eos_token(ctx: ChatGLMContext):
    if False:
        while True:
            i = 10
    return ctx.pipeline.model.config.eos_token_id