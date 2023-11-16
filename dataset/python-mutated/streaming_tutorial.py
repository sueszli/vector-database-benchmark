from typing import List
import asyncio
import logging
from queue import Empty
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from ray import serve
logger = logging.getLogger('ray.serve')
fastapi_app = FastAPI()

@serve.deployment
@serve.ingress(fastapi_app)
class Textbot:

    def __init__(self, model_id: str):
        if False:
            while True:
                i = 10
        self.loop = asyncio.get_running_loop()
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @fastapi_app.post('/')
    def handle_request(self, prompt: str) -> StreamingResponse:
        if False:
            i = 10
            return i + 15
        logger.info(f'Got prompt: "{prompt}"')
        streamer = TextIteratorStreamer(self.tokenizer, timeout=0, skip_prompt=True, skip_special_tokens=True)
        self.loop.run_in_executor(None, self.generate_text, prompt, streamer)
        return StreamingResponse(self.consume_streamer(streamer), media_type='text/plain')

    def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
        if False:
            for i in range(10):
                print('nop')
        input_ids = self.tokenizer([prompt], return_tensors='pt').input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=10000)

    async def consume_streamer(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    logger.info(f'Yielding token: "{token}"')
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)
app = Textbot.bind('microsoft/DialoGPT-small')
serve.run(app)
chunks = []
import requests
prompt = 'Tell me a story about dogs.'
response = requests.post(f'http://localhost:8000/?prompt={prompt}', stream=True)
response.raise_for_status()
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end='')
    chunks.append(chunk)
assert chunks == ['Dogs ', 'are ', 'the ', 'best.']
import asyncio
import logging
from queue import Empty
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from ray import serve
logger = logging.getLogger('ray.serve')
fastapi_app = FastAPI()

@serve.deployment
@serve.ingress(fastapi_app)
class Chatbot:

    def __init__(self, model_id: str):
        if False:
            i = 10
            return i + 15
        self.loop = asyncio.get_running_loop()
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @fastapi_app.websocket('/')
    async def handle_request(self, ws: WebSocket) -> None:
        await ws.accept()
        conversation = ''
        try:
            while True:
                prompt = await ws.receive_text()
                logger.info(f'Got prompt: "{prompt}"')
                conversation += prompt
                streamer = TextIteratorStreamer(self.tokenizer, timeout=0, skip_prompt=True, skip_special_tokens=True)
                self.loop.run_in_executor(None, self.generate_text, conversation, streamer)
                response = ''
                async for text in self.consume_streamer(streamer):
                    await ws.send_text(text)
                    response += text
                await ws.send_text('<<Response Finished>>')
                conversation += response
        except WebSocketDisconnect:
            print('Client disconnected.')

    def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
        if False:
            for i in range(10):
                print('nop')
        input_ids = self.tokenizer([prompt], return_tensors='pt').input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=10000)

    async def consume_streamer(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    logger.info(f'Yielding token: "{token}"')
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)
app = Chatbot.bind('microsoft/DialoGPT-small')
serve.run(app)
chunks = []
(original_print, print) = (print, lambda chunk, end=None: chunks.append(chunk))
from websockets.sync.client import connect
with connect('ws://localhost:8000') as websocket:
    websocket.send('Space the final')
    while True:
        received = websocket.recv()
        if received == '<<Response Finished>>':
            break
        print(received, end='')
    print('\n')
    websocket.send(' These are the voyages')
    while True:
        received = websocket.recv()
        if received == '<<Response Finished>>':
            break
        print(received, end='')
    print('\n')
assert chunks == [' ', '', '', 'frontier.', '\n', ' ', 'of ', 'the ', 'starship ', '', '', 'Enterprise.', '\n']
print = original_print
import asyncio
import logging
from queue import Empty, Queue
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from ray import serve
logger = logging.getLogger('ray.serve')

class RawStreamer:

    def __init__(self, timeout: float=None):
        if False:
            for i in range(10):
                print('nop')
        self.q = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, values):
        if False:
            while True:
                i = 10
        self.q.put(values)

    def end(self):
        if False:
            for i in range(10):
                print('nop')
        self.q.put(self.stop_signal)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __next__(self):
        if False:
            return 10
        result = self.q.get(timeout=self.timeout)
        if result == self.stop_signal:
            raise StopIteration()
        else:
            return result
fastapi_app = FastAPI()

@serve.deployment
@serve.ingress(fastapi_app)
class Batchbot:

    def __init__(self, model_id: str):
        if False:
            return 10
        self.loop = asyncio.get_running_loop()
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @fastapi_app.post('/')
    async def handle_request(self, prompt: str) -> StreamingResponse:
        logger.info(f'Got prompt: "{prompt}"')
        return StreamingResponse(self.run_model(prompt), media_type='text/plain')

    @serve.batch(max_batch_size=2, batch_wait_timeout_s=15)
    async def run_model(self, prompts: List[str]):
        streamer = RawStreamer()
        self.loop.run_in_executor(None, self.generate_text, prompts, streamer)
        on_prompt_tokens = True
        async for decoded_token_batch in self.consume_streamer(streamer):
            if not on_prompt_tokens:
                logger.info(f'Yielding decoded_token_batch: {decoded_token_batch}')
                yield decoded_token_batch
            else:
                logger.info(f'Skipped prompts: {decoded_token_batch}')
                on_prompt_tokens = False

    def generate_text(self, prompts: str, streamer: RawStreamer):
        if False:
            print('Hello World!')
        input_ids = self.tokenizer(prompts, return_tensors='pt', padding=True).input_ids
        self.model.generate(input_ids, streamer=streamer, max_length=10000)

    async def consume_streamer(self, streamer: RawStreamer):
        while True:
            try:
                for token_batch in streamer:
                    decoded_tokens = []
                    for token in token_batch:
                        decoded_tokens.append(self.tokenizer.decode(token, skip_special_tokens=True))
                    logger.info(f'Yielding decoded tokens: {decoded_tokens}')
                    yield decoded_tokens
                break
            except Empty:
                await asyncio.sleep(0.001)
app = Batchbot.bind('microsoft/DialoGPT-small')
serve.run(app)
from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor

def get_buffered_response(prompt) -> List[str]:
    if False:
        print('Hello World!')
    response = requests.post(f'http://localhost:8000/?prompt={prompt}', stream=True)
    chunks = []
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        chunks.append(chunk)
    return chunks
with ThreadPoolExecutor() as pool:
    futs = [pool.submit(partial(get_buffered_response, prompt)) for prompt in ['Introduce yourself to me!', 'Tell me a story about dogs.']]
    responses = [fut.result() for fut in futs]
    assert responses == [['I', "'m", ' not', ' sure', ' if', ' I', "'m", ' ready', ' for', ' that', '.'], ['D', 'ogs', ' are', ' the', ' best', '.']]