import json
from typing import AsyncGenerator
from fastapi import BackgroundTasks
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from ray import serve

@serve.deployment(ray_actor_options={'num_gpus': 1})
class VLLMPredictDeployment:

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a VLLM deployment.\n\n        Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py\n        for the full list of arguments.\n\n        Args:\n            model: name or path of the huggingface model to use\n            download_dir: directory to download and load the weights,\n                default to the default cache dir of huggingface.\n            use_np_weights: save a numpy copy of model weights for\n                faster loading. This can increase the disk usage by up to 2x.\n            use_dummy_weights: use dummy values for model weights.\n            dtype: data type for model weights and activations.\n                The "auto" option will use FP16 precision\n                for FP32 and FP16 models, and BF16 precision.\n                for BF16 models.\n            seed: random seed.\n            worker_use_ray: use Ray for distributed serving, will be\n                automatically set when using more than 1 GPU\n            pipeline_parallel_size: number of pipeline stages.\n            tensor_parallel_size: number of tensor parallel replicas.\n            block_size: token block size.\n            swap_space: CPU swap space size (GiB) per GPU.\n            gpu_memory_utilization: the percentage of GPU memory to be used for\n                the model executor\n            max_num_batched_tokens: maximum number of batched tokens per iteration\n            max_num_seqs: maximum number of sequences per iteration.\n            disable_log_stats: disable logging statistics.\n            engine_use_ray: use Ray to start the LLM engine in a separate\n                process as the server process.\n            disable_log_requests: disable logging requests.\n        '
        args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {'text': text_output}
            yield (json.dumps(ret) + '\n').encode('utf-8')
            num_returned += len(text_output)

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Response:
        """Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        """
        request_dict = await request.json()
        prompt = request_dict.pop('prompt')
        stream = request_dict.pop('stream', False)
        sampling_params = SamplingParams(**request_dict)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        if stream:
            background_tasks = BackgroundTasks()
            background_tasks.add_task(self.may_abort_request, request_id)
            return StreamingResponse(self.stream_results(results_generator), background=background_tasks)
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                await self.engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output
        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {'text': text_outputs}
        return Response(content=json.dumps(ret))

def send_sample_request():
    if False:
        print('Hello World!')
    import requests
    prompt = 'How do I cook fried rice?'
    sample_input = {'prompt': prompt, 'stream': True}
    output = requests.post('http://localhost:8000/', json=sample_input)
    for line in output.iter_lines():
        print(line.decode('utf-8'))
if __name__ == '__main__':
    deployment = VLLMPredictDeployment.bind(model='facebook/opt-125m')
    serve.run(deployment)
    send_sample_request()