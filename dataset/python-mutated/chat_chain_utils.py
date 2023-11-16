import json
import re
from typing import Callable
import requests
import transformers
from chat_chain_prompts import INSTRUCTIONS, OBSERVATION_SEQ, TOOLS_PREFIX
from hf_langchain_inference import HFInference
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from loguru import logger
from oasst_shared.schemas import inference
from openapi_parser import prepare_plugin_for_llm
from settings import settings
from utils import shared_tokenizer_lock, special_tokens
RESPONSE_MAX_LENGTH = 2048
DESCRIPTION_FOR_MODEL_MAX_LENGTH = 512
llm_json_parser = HFInference(inference_server_url=settings.inference_server_url, max_new_tokens=512, stop_sequences=[special_tokens['end'] if special_tokens['end'] else '</s>'], top_k=5, temperature=0.2, repetition_penalty=1 / 0.83)

def similarity(ts1: str, ts2: str) -> float:
    if False:
        i = 10
        return i + 15
    'Compute Jaro-Winkler distance between two strings.'
    if ts1 == ts2:
        return 1
    match = 0
    (len1, len2) = (len(ts1), len(ts2))
    max_dist = max(len1, len2) // 2 - 1
    hash_ts1 = [0] * len1
    hash_ts2 = [0] * len2
    for i in range(len1):
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):
            if ts1[i] == ts2[j] and hash_ts2[j] == 0:
                hash_ts1[i] = 1
                hash_ts2[j] = 1
                match += 1
                break
    if match == 0:
        return 0
    t = 0
    point = 0
    for i in range(len1):
        if hash_ts1[i] == 1:
            while hash_ts2[point] == 0:
                point += 1
            if ts1[i] != ts2[point]:
                t += 1
            point += 1
    t /= 2
    return (match / len1 + match / len2 + (match - t) / match) / 3.0

def extract_tool_and_input(llm_output: str, ai_prefix: str) -> tuple[str, str]:
    if False:
        while True:
            i = 10
    '\n    Extract tool name and tool input from LLM output. If LLM chose not to use a tool, `ai_prefix` is returned instead of tool name, and LLM output is returned instead of tool input.\n    '
    llm_output = llm_output.strip().replace('```', '')
    if f'{ai_prefix}:' in llm_output:
        return (ai_prefix, llm_output.split(f'{ai_prefix}:')[-1].strip())
    regex = 'Action: (.*?)[\\n]*Action Input:\\n?(.*)'
    match = re.search(regex, llm_output, re.MULTILINE | re.DOTALL)
    if not match:
        if OBSERVATION_SEQ in llm_output:
            return (ai_prefix, llm_output.split(OBSERVATION_SEQ)[-1].strip())
        return (ai_prefix, llm_output)
    action = match.group(1)
    action_input = match.group(2)
    return (action.strip().replace("'", ''), action_input.strip().strip(' '))

def truncate_str(output: str, max_length: int=1024) -> str:
    if False:
        i = 10
        return i + 15
    if len(output) > max_length:
        if output[0] == '(':
            return output[:max_length] + '...)'
        elif output[0] == '[':
            return output[:max_length] + '...]'
        elif output[0] == '{':
            return output[:max_length] + '...}'
        else:
            return output[:max_length] + '...'
    return output

def prepare_json(json_str: str) -> str:
    if False:
        return 10
    json_str = json_str.strip()
    fixed_json = json_str
    try:
        json.loads(json_str)
    except json.decoder.JSONDecodeError:
        fixed_json = re.sub('(?<=\\{|\\,)(\\s*)(\\w+)(\\s*):', '\\1"\\2"\\3:', json_str)
        fixed_json = fixed_json.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        brace_count = bracket_count = 0
        result = []
        for c in fixed_json:
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
            elif c == '[':
                bracket_count += 1
            elif c == ']':
                bracket_count -= 1
            if brace_count >= 0 and bracket_count >= 0:
                result.append(c)
        result.extend(['}'] * brace_count)
        result.extend([']'] * bracket_count)
        fixed_json = ''.join(result)
        try:
            json.loads(fixed_json)
        except json.decoder.JSONDecodeError as e:
            logger.warning(f'JSON is still not valid, trying to fix it with LLM {fixed_json}')
            prompt = f"{special_tokens['prompter']}Below is malformed JSON object string:\n--------------\n{json_str}\n--------------\nParsing error:\n--------------\n{e}\n\nRULES:\n1. If malformed JSON object string contains multiple objects, you merge them into one.\n2. You will never made up or add any new data, you will only fix the malformed JSON object string.\n\nHere is the fixed JSON object string:{special_tokens['end'] or '</s>'}{special_tokens['assistant']}"
            logger.warning(f'JSON Fix Prompt: {prompt}')
            out = llm_json_parser.generate(prompts=[prompt]).generations[0][0].text
            out = out[:out.find('}') + 1]
            logger.warning(f'JSON Fix Output: {out}')
            return out
    return fixed_json

def select_tool(tool_name: str, tools: list[Tool]) -> Tool | None:
    if False:
        return 10
    tool = next((t for t in tools if t.name in tool_name), None)
    if tool:
        return tool
    (tool, tool_similarity) = max(((t, similarity(t.name, tool_name)) for t in tools), key=lambda x: x[1], default=(None, 0))
    if tool and tool_similarity > 0.75:
        return tool
    return None

def use_tool(tool_name: str, tool_input: str, tools: list[Tool]) -> str:
    if False:
        for i in range(10):
            print('nop')
    tool = select_tool(tool_name, tools)
    if not tool:
        return f'ERROR! {tool_name} is not a valid tool. Try again with different tool!'
    prepared_input = prepare_json(tool_input)
    tool_output = tool.func(prepared_input)
    return tool_output

class RequestsForLLM:

    def run(self, params: str, url: str, param_location: str, type: str, payload: str | None=None) -> str:
        if False:
            while True:
                i = 10
        return self.run_request(params, url, param_location, type, payload)

    def run_request(self, params: str, url: str, param_location: str, type: str, payload: str=None) -> str:
        if False:
            print('Hello World!')
        try:
            query_params = params
            if param_location == 'path':
                for (key, value) in query_params.items():
                    url = url.replace(f'{{{key}}}', value)
                query_params = {}
            headers = {'Content-Type': 'application/json'} if payload else None
            if type.lower() == 'get':
                logger.info(f'Running {type.upper()} request on {url} with\nparams: {params}\nparam_location: {param_location}\npayload: {payload}')
                res = requests.get(url, params=query_params, headers=headers)
            elif type.lower() == 'post':
                data = json.dumps(payload) if payload else json.dumps(params)
                logger.info(f'Running {type.upper()} request on {url} with\nparams: {params}\nparam_location: {param_location}\npayload: {data}')
                res = requests.post(url, params=query_params, data=data, headers=headers)
            else:
                return f'ERROR! Unsupported request type: {type}. Only GET and POST are supported. Try again!'
            return self.process_response(res)
        except Exception as e:
            return f"ERROR! That didn't work, try modifying Action Input.\n{e}. Try again!"

    def process_response(self, res: requests.Response) -> str:
        if False:
            i = 10
            return i + 15
        logger.info(f'Request response: {res.text}')
        if res.status_code != 200:
            return f'ERROR! Please modify Action Input. according to this error message: \n{res.text}. Try again!'
        if res.text is None or len(res.text) == 0:
            return "ERROR! That didn't work, try modifying Action Input.\nEmpty response. Try again!"
        if 'null' in res.text.lower() and len(res.text) < 10:
            return "ERROR! That didn't work, try modifying Action Input.\nEmpty response. Try again!"
        return truncate_str(res.text, RESPONSE_MAX_LENGTH)

def compose_tools_from_plugin(plugin: inference.PluginEntry | None) -> tuple[str, list[Tool]]:
    if False:
        i = 10
        return i + 15
    if not plugin:
        return ('', [])
    llm_plugin: inference.PluginConfig = prepare_plugin_for_llm(plugin.url)
    if not llm_plugin:
        return ('', [])
    tools = []
    request_tool = RequestsForLLM()

    def create_tool_func(endpoint: inference.PluginOpenAPIEndpoint, param_location: str) -> Callable[..., str]:
        if False:
            print('Hello World!')

        def func(req) -> str:
            if False:
                print('Hello World!')
            try:
                json_obj = json.loads(req)
                request = json_obj.get('request', {})
                params = request.get('params', {})
                payload = request.get('payload', None)
            except json.JSONDecodeError:
                print('Error: Invalid JSON input')
                (request, params, payload) = ({}, {}, None)
            except Exception as e:
                print(f'Error: {e}')
                (request, params, payload) = ({}, {}, None)
            return request_tool.run(url=endpoint.url, params=params, param_location=param_location, type=endpoint.type, payload=payload)
        return func
    for endpoint in llm_plugin.endpoints:
        params = '\n\n'.join([f' name: "{param.name}",\n in: "{param.in_}",\n description: "{truncate_str(param.description, 128)}",\n schema: {param.schema_},\n required: {param.required}' for param in endpoint.params])
        params = params.replace('{', '{{').replace('}', '}}')
        payload_description = ''
        if endpoint.payload:
            try:
                payload_description = 'payload: ' + truncate_str(json.dumps(endpoint.payload, indent=4), 256)
                payload_description = payload_description.replace('{', '{{').replace('}', '}}')
            except Exception as e:
                logger.warning(f'Failed to convert payload to json string: {e}')
        payload_description += '' if not payload_description or payload_description.endswith('\n') else '\n'
        if len(payload_description) > 0:
            payload_description = '\n' + payload_description + '\n'
        parameters_description = f'params:\n{params}\n' if params else '\n'
        openapi_specification_title = '\nOpenAPI specification\n' if len(payload_description) > 0 or len(params) > 0 else ''
        param_location = endpoint.params[0].in_ if len(endpoint.params) > 0 else 'query'
        path = endpoint.path[1:] if endpoint.path and len(endpoint.path) > 0 else endpoint.path
        tool = Tool(name=endpoint.operation_id if endpoint.operation_id != '' else path, func=create_tool_func(endpoint, param_location), description=f'{openapi_specification_title}{parameters_description}{payload_description}tool description: {endpoint.summary}\n')
        tools.append(tool)
    tools_string = '\n'.join([f'> {tool.name}{tool.description}' for tool in tools])
    plugin_description_for_model = truncate_str(llm_plugin.description_for_model, DESCRIPTION_FOR_MODEL_MAX_LENGTH)
    return (f'{TOOLS_PREFIX}{tools_string}\n\n{llm_plugin.name_for_model} plugin description:\n{plugin_description_for_model}\n\n{INSTRUCTIONS}', tools)

def prepare_prompt(input_prompt: str, prompt_template: PromptTemplate, memory: ConversationBufferMemory, tools_names: list[str] | None, current_time: str, language: str, tokenizer: transformers.PreTrainedTokenizer, worker_config: inference.WorkerConfig, action_input_format: str, custom_instructions: str='') -> str:
    if False:
        i = 10
        return i + 15
    max_input_length = worker_config.model_config.max_input_length
    args = {'input': input_prompt, 'language': language, 'current_time': current_time, 'chat_history': memory.buffer, 'custom_instructions': custom_instructions}
    if tools_names:
        args['tools_names'] = tools_names
        args['action_input_format'] = action_input_format
    out_prompt = prompt_template.format(**args)
    with shared_tokenizer_lock:
        ids = tokenizer.encode(out_prompt)
    while len(ids) > max_input_length and len(memory.chat_memory.messages) > 0:
        memory.chat_memory.messages.pop(0)
        args = {'input': input_prompt, 'language': language, 'current_time': current_time, 'chat_history': memory.buffer, 'custom_instructions': custom_instructions}
        if tools_names:
            args['tools_names'] = tools_names
            args['action_input_format'] = action_input_format
        out_prompt = prompt_template.format(**args)
        with shared_tokenizer_lock:
            ids = tokenizer.encode(out_prompt)
        logger.warning(f'Prompt too long, deleting chat history. New length: {len(ids)}')
    return out_prompt