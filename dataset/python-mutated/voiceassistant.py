from langchain import LLMChain, PromptTemplate
from bigdl.llm.langchain.llms import *
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import speech_recognition as sr
import pyttsx3
import argparse

def prepare_chain(args):
    if False:
        for i in range(10):
            print('nop')
    model_path = args.model_path
    n_threads = args.thread_num
    n_ctx = args.context_size
    template = '\n    {history}\n\n    Q: {human_input}\n    A:'
    prompt = PromptTemplate(input_variables=['history', 'human_input'], template=template)
    model_family_to_llm = {'llama': LlamaLLM, 'gptneox': GptneoxLLM, 'bloom': BloomLLM, 'starcoder': StarcoderLLM, 'chatglm': ChatGLMLLM}
    if model_family in model_family_to_llm:
        langchain_llm = model_family_to_llm[model_family]
    else:
        raise ValueError(f'Unknown model family: {model_family}')
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = langchain_llm(model_path=model_path, n_threads=n_threads, callback_manager=callback_manager, verbose=True, n_ctx=n_ctx, stop=['\n\n'])
    voiceassitant_chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=ConversationBufferWindowMemory(k=2))
    return voiceassitant_chain

def listen(voiceassitant_chain):
    if False:
        print('Hello World!')
    engine = pyttsx3.init()
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Calibrating...')
        r.adjust_for_ambient_noise(source, duration=5)
        print('Okay, go!')
        while 1:
            text = ''
            print('listening now...')
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=30)
                print('Recognizing...')
                text = r.recognize_whisper(audio, model='medium.en', show_dict=True)['text']
            except Exception as e:
                unrecognized_speech_text = f"Sorry, I didn't catch that. Exception was: {e}s"
                text = unrecognized_speech_text
            print(text)
            response_text = voiceassitant_chain.predict(human_input=text)
            print(response_text)
            engine.say(response_text)
            engine.runAndWait()

def main(args):
    if False:
        i = 10
        return i + 15
    chain = prepare_chain(args)
    listen(chain)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigDLCausalLM Langchain Voice Assistant Example')
    parser.add_argument('-x', '--model-family', type=str, required=True, choices=['llama', 'bloom', 'gptneox', 'chatglm', 'starcoder'], help='the model family')
    parser.add_argument('-m', '--model-path', type=str, required=True, help='the path to the converted llm model')
    parser.add_argument('-t', '--thread-num', type=int, default=2, help='Number of threads to use for inference')
    parser.add_argument('-c', '--context-size', type=int, default=512, help='Maximum context size')
    args = parser.parse_args()
    main(args)