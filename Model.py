import gc
import time
import torch
from gtts import gTTS
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

MODEL_PATH = '/media/shivek/DATA/Studies/Projects/Manimatic/LLMs/llama-2-7b-chat.Q5_K_M.gguf'
MODEL_PATH_02 = '/media/shivek/DATA/Studies/Projects/Manimatic/LLMs/llama-2-13b-chat.Q3_K_M.gguf'

def create_prompt() -> PromptTemplate:
    
    _DEFAULT_TEMPLATE: str = """You have immense knowledge of mathematics.
    You will explain all mathematical concepts in a very detailed manner, covering from the basics to the advanced.
    You will explain the concept in the following way:
    First, You will explain the logic behind the concept in detail. 
    Then you will provide an example for it.
    Second, You will write the official definition pertaining to the concept.
    You will explain the definition in detail.
    You will then give an example for it.
    Third, You will list down the key points regarding the concept.
    Fourth, You will then explain each key point in detail.
    Fifth, You will give 2 examples for each key point, and explain each example pertaining to the concept.
    Give a different example each time, and examples should not be repeated.
    You will not repeat a line more than twice to avoid looping.
    Then, you will conclude the explanation.
    You will stop as soon as you have given all the relevant explanation.
                                                            
    Question:{question}
    Answer:"""
    
    prompt: PromptTemplate = PromptTemplate(input_variables = ['question'], template = _DEFAULT_TEMPLATE)
    
    return prompt

def create_prompt_02() -> PromptTemplate:
    
    _DEFAULT_TEMPLATE_02: str = """You can visualise mathematics based on a given information.
    You will provide me with an explanation that will help me understand how to visualise every aspect of the given information.
    The explain should be clear and in detail, for excellent understanding.
                                                            
    Query:{query}
    Answer:"""
    
    prompt_02: PromptTemplate = PromptTemplate(input_variables = ['query'], template = _DEFAULT_TEMPLATE_02)
    
    return prompt_02

def create_prompt_03() -> PromptTemplate:

    _DEFAULT_TEMPLATE_03: str = """You are a professional programmer.
    You will write a manim script for the information provided.
    The output should only contain the code.
    The code should explain everything of the given information.
    The manim code should be of the manim community version 0.14.0.
    You will stop generating     as soon as you have provided the relevant code. 
    Query:{query}
    Code:"""
   
    prompt_03: PromptTemplate = PromptTemplate(input_variables = ['query'], template = _DEFAULT_TEMPLATE_03)
    
    return prompt_03

def load_model() -> LLMChain:

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    Llama_model = LlamaCpp(model_path = MODEL_PATH, temperature = 0.4, n_gpu_layers = 6, n_batch = 64, max_tokens = 4096, top_p = 0.9, callback_manager = callback_manager, verbose = True, n_ctx = 16384)
    
    prompt: PromptTemplate = create_prompt()
    
    llm_chain = LLMChain(llm = Llama_model, prompt = prompt)
    
    return llm_chain

llm_chain = load_model()

model_prompt: str = input('\n\nEnter your prompt:') + '\n\n'
response: str = llm_chain.invoke(model_prompt)
data = response
text = data['text'].strip()
cleaned_response = text

gc.collect()
torch.cuda.empty_cache()
time.sleep(10)

def load_model_02() -> LLMChain:

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    Llama_model_02 = LlamaCpp(model_path = MODEL_PATH, temperature = 0.7, n_gpu_layers = 6, n_batch = 64, max_tokens = 4096, top_p = 0.9, callback_manager = callback_manager, verbose = True, n_ctx = 16384)
    
    prompt_02: PromptTemplate = create_prompt_02()
    
    llm_chain_02 = LLMChain(llm = Llama_model_02, prompt = prompt_02)
    
    return llm_chain_02

llm_chain_02 = load_model_02()

model_prompt_02: str = cleaned_response + '\n\n' + 'Explain me how to visualise each and every part of this explanation in detail.'
response_02: str = llm_chain_02.invoke(model_prompt_02)
data_02 = response_02
text_02 = data_02['text'].strip()
cleaned_response_02 = text_02

gc.collect()
torch.cuda.empty_cache()
time.sleep(10)

def load_model_03() -> LLMChain:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    Llama_model_03 = LlamaCpp(model_path = MODEL_PATH, temperature = 0.3, n_gpu_layers = 6, n_batch = 64, max_tokens = 4096, top_p = 0.9, callback_manager = callback_manager, verbose = True, n_ctx = 16384)
    
    prompt_03: PromptTemplate = create_prompt_03()
    
    llm_chain_03 = LLMChain(llm = Llama_model_03, prompt = prompt_03)
    
    return llm_chain_03

llm_chain_03 = load_model_03()

model_prompt_03: str = cleaned_response + cleaned_response_02 + '\n\n' + 'Write manim code that can visualise the given context.'
response_03: str = llm_chain_03.invoke(model_prompt_03)
data_03 = response_03
text_03 = data_03['text'].strip()
cleaned_response_03 = text_03

gc.collect()
torch.cuda.empty_cache()
time.sleep(10)

exec(cleaned_response_03, globals = None, locals = None)

response_tts = cleaned_response
language = 'en'
tts = gTTS(text = response_tts, lang = language)
audio_file = f"{model_prompt}.mp3"
tts.save(audio_file)
print(f"Audio saved to: {audio_file}\n\n")