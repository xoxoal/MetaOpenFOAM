
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.embeddings.openai import OpenAIEmbeddings
import os

import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import RetrievalQA

from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import config_path
from langchain_community.callbacks import get_openai_callback
from Statistics import global_statistics

import aiohttp
import base64
import sys


class AsyncQA_tutorial:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_tutorial, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_tutorial()
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, question):
        loop = asyncio.get_running_loop()

        result, usage_info = await loop.run_in_executor(self.executor, self._execute_with_callback, question)

        global_statistics.total_tokens += usage_info.get('total_tokens', 0)
        global_statistics.prompt_tokens += usage_info.get('prompt_tokens', 0)
        global_statistics.completion_tokens += usage_info.get('completion_tokens', 0)
        return result
    
    def _execute_with_callback(self, question):
        with get_openai_callback() as cb:
            result = self.qa_interface(question)
            usage_info = {
                'total_tokens': cb.total_tokens,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_cost': cb.total_cost,
            }
            return result, usage_info

    def close(self):
        self.executor.shutdown()

class AsyncQA_tutorial_name:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_tutorial_name, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_tutorial_name()
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, question):
        loop = asyncio.get_running_loop()

        result, usage_info = await loop.run_in_executor(self.executor, self._execute_with_callback, question)

        global_statistics.total_tokens += usage_info.get('total_tokens', 0)
        global_statistics.prompt_tokens += usage_info.get('prompt_tokens', 0)
        global_statistics.completion_tokens += usage_info.get('completion_tokens', 0)
        return result
    def _execute_with_callback(self, question):
        with get_openai_callback() as cb:
            result = self.qa_interface(question)
            usage_info = {
                'total_tokens': cb.total_tokens,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_cost': cb.total_cost,
            }
            return result, usage_info

    def close(self):
        self.executor.shutdown()

class AsyncQA_allrun:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_allrun, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_allrun()
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, question):
        loop = asyncio.get_running_loop()
        result, usage_info = await loop.run_in_executor(self.executor, self._execute_with_callback, question)

        global_statistics.total_tokens += usage_info.get('total_tokens', 0)
        global_statistics.prompt_tokens += usage_info.get('prompt_tokens', 0)
        global_statistics.completion_tokens += usage_info.get('completion_tokens', 0)
        return result
    def _execute_with_callback(self, question):
        with get_openai_callback() as cb:
            result = self.qa_interface(question)
            usage_info = {
                'total_tokens': cb.total_tokens,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_cost': cb.total_cost,
            }
            return result, usage_info

    def close(self):
        self.executor.shutdown()

class AsyncQA_command_help:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_command_help, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_command_help()
            self.executor = ThreadPoolExecutor()
            self._initialized = True
            
    async def ask(self, question):
        loop = asyncio.get_running_loop()
        # 在线程池中运行同步函数

        result, usage_info = await loop.run_in_executor(self.executor, self._execute_with_callback, question)

        global_statistics.total_tokens += usage_info.get('total_tokens', 0)
        global_statistics.prompt_tokens += usage_info.get('prompt_tokens', 0)
        global_statistics.completion_tokens += usage_info.get('completion_tokens', 0)
        return result
    def _execute_with_callback(self, question):
        with get_openai_callback() as cb:
            result = self.qa_interface(question)
            usage_info = {
                'total_tokens': cb.total_tokens,
                'prompt_tokens': cb.prompt_tokens,
                'completion_tokens': cb.completion_tokens,
                'total_cost': cb.total_cost,
            }
            return result, usage_info

    def close(self):
        self.executor.shutdown()

class AsyncQA_Ori:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncQA_Ori, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def init_instance(self):
        if not self._initialized:
            self.qa_interface = setup_qa_ori()
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, question):
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(self.executor, self.qa_interface, question)
        return result

    def close(self):
        self.executor.shutdown()

class AsyncImageQA:
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AsyncImageQA, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def init_instance(self):
        if not self._initialized:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            self.executor = ThreadPoolExecutor()
            self._initialized = True

    async def ask(self, image_path: str, question: str):
        """
        异步调用 GPT-4o API 进行图片问答。

        Args:
            image_path (str): 图片路径。
            question (str): 关于图片的问题。

        Returns:
            dict: 包含 GPT 回复和 token 使用情况。
        """
        base64_image = await self._encode_image(image_path)
        payload = self._construct_payload(base64_image, question)

        async with aiohttp.ClientSession() as session:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            async with session.post(
                "https://api.openai-proxy.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                try:
                    response_data = await response.json()
                    print("Response data:", response_data)
                except Exception as e:
                    print("Error reading response:", e)
                    print("Response status:", response.status)

                if response.status == 200:
                    reply = response_data.get('choices', [{}])[0].get('message', {}).get('content', "No reply received.")
                    #total_tokens = response_data.get('usage', {}).get('total_tokens', 0)
                    usage = response_data.get('usage', {})
                    total_tokens = usage.get('total_tokens', 0)
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    # 更新全局统计
                    global_statistics.total_tokens += total_tokens
                    global_statistics.prompt_tokens += prompt_tokens
                    global_statistics.completion_tokens += completion_tokens
                    
                    return {"reply": reply, "total_tokens": total_tokens}
                else:
                    error_message = response_data.get('error', {}).get('message', 'Unknown error')
                    return {"error": error_message, "total_tokens": 0}

    async def _encode_image(self, image_path):
        """
        异步读取并编码图片为 Base64。

        Args:
            image_path (str): 图片路径。

        Returns:
            str: Base64 编码的图片数据。
        """
        loop = asyncio.get_running_loop()
        with open(image_path, "rb") as image_file:
            return await loop.run_in_executor(self.executor, lambda: base64.b64encode(image_file.read()).decode('utf-8'))

    def _construct_payload(self, base64_image: str, question: str):
        """
        构造 API 请求的 payload。

        Args:
            base64_image (str): Base64 编码的图片。
            question (str): 关于图片的问题。

        Returns:
            dict: API 请求的 payload。
        """
        return {
            "model": config_path.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10000,
            "temperature": config_path.temperature
        }

    def close(self):
        """
        关闭线程池。
        """
        self.executor.shutdown()


def setup_qa_ori():
    if config_path.model.lower().startswith("deepseek"):
        # 假设 ChatDeepSeek 已经实现，并支持直接调用，且不需要额外的 API Key
        deepseek_model = ChatDeepSeek(model=config_path.model, temperature=config_path.temperature)
        
        def get_deepseek_response(user_msg):
            # 调用 DeepSeek 模型，获取响应结果
            chat_completion = deepseek_model(user_msg)
            chat_completion_dict = dict(chat_completion)
            print("DeepSeek response keys:", chat_completion_dict.keys())
            
            # 统计 token 使用情况
            # usage = dict(chat_completion_dict['usage'])
            usage = chat_completion_dict.get('usage_metadata', {})
            print("DeepSeek usage keys:", usage.keys())
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
            print('input_tokens:',input_tokens)
            print('output_tokens:',output_tokens)
            print('total_tokens:',total_tokens)
            # 更新全局统计（单位：token）
            global_statistics.total_tokens += total_tokens
            global_statistics.prompt_tokens += input_tokens
            global_statistics.completion_tokens += output_tokens
            # prompt_tokens = usage.get('prompt_tokens', 0)
            # completion_tokens = usage.get('completion_tokens', 0)
            # total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            # print('total_tokens:',total_tokens)
            # # 如果存在思维链 token 数，则也可以获取（可选）
            # reasoning_tokens = 0
            # if 'completion_tokens_details' in usage:
            #     details = usage['completion_tokens_details']
            #     reasoning_tokens = details.get('reasoning_tokens', 0)

            # global_statistics.total_tokens += total_tokens
            # global_statistics.prompt_tokens += prompt_tokens
            # global_statistics.completion_tokens += completion_tokens
            #print('chat_completion.content:',chat_completion.content)
            # 返回响应内容，格式与 GPT 返回一致
            return chat_completion.content
        
        return get_deepseek_response
    
    else:
        def get_gpt4o_response(user_msg):

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": user_msg
                    }
                ],
                model=config_path.model,
                temperature=config_path.temperature
            )
            chat_completion_dict = dict(chat_completion)
            print(chat_completion_dict.keys())
            usage = chat_completion_dict['usage']
            usage = dict(usage)
            total_tokens = usage['total_tokens']
            prompt_tokens = usage['prompt_tokens']
            completion_tokens = usage['completion_tokens']

            global_statistics.total_tokens += total_tokens
            global_statistics.prompt_tokens += prompt_tokens
            global_statistics.completion_tokens += completion_tokens

            return chat_completion.choices[0].message.content

    return get_gpt4o_response

def setup_qa_tutorial():

    persist_directory = f'{config_path.Database_PATH}/openfoam_tutorials'
    # vectordb = FAISS.load_local(persist_directory, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    # chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    vectordb = FAISS.load_local(persist_directory, HuggingFaceEmbeddings(),allow_dangerous_deserialization=True)
    if config_path.model.lower().startswith("deepseek"):
        chat_model = ChatDeepSeek(model=config_path.model, temperature=config_path.temperature)
    elif config_path.model.lower().startswith("gpt"):
        chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": config_path.searchdocs})

    qa_interface = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


    return qa_interface

def setup_qa_tutorial_name():

    persist_directory = f'{config_path.Database_PATH}/openfoam_tutorials_summary'
    # vectordb = FAISS.load_local(persist_directory, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    # chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    vectordb = FAISS.load_local(persist_directory, HuggingFaceEmbeddings(),allow_dangerous_deserialization=True)
    if config_path.model.lower().startswith("deepseek"):
        chat_model = ChatDeepSeek(model=config_path.model, temperature=config_path.temperature)
    elif config_path.model.lower().startswith("gpt"):
        chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": config_path.searchdocs})
    qa_interface = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_interface

def setup_qa_allrun():

    persist_directory = f'{config_path.Database_PATH}/openfoam_allrun'
    # vectordb = FAISS.load_local(persist_directory, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    # chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    vectordb = FAISS.load_local(persist_directory, HuggingFaceEmbeddings(),allow_dangerous_deserialization=True)
    if config_path.model.lower().startswith("deepseek"):
        chat_model = ChatDeepSeek(model=config_path.model, temperature=config_path.temperature)
    elif config_path.model.lower().startswith("gpt"):
        chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": config_path.searchdocs})
    qa_interface = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_interface

def setup_qa_command_help():

    persist_directory = f'{config_path.Database_PATH}/openfoam_command_helps'
    # vectordb = FAISS.load_local(persist_directory, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
    vectordb = FAISS.load_local(persist_directory, HuggingFaceEmbeddings(),allow_dangerous_deserialization=True)
    if config_path.model.lower().startswith("deepseek"):
        chat_model = ChatDeepSeek(model=config_path.model, temperature=config_path.temperature)
    elif config_path.model.lower().startswith("gpt"):
        chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    else:
        print('the chat model is not identified, only the model start with deepseek and gpt can be identified.')
        sys.exit()
    #chat_model = ChatDeepSeek(model="deepseek-chat", temperature=0.01)
    #chat_model = ChatOpenAI(model=config_path.model, temperature=config_path.temperature)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": config_path.searchdocs})
    qa_interface = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_interface
