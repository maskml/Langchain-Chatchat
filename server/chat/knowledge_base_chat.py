from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS, 
                     VECTOR_SEARCH_TOP_K, 
                     SCORE_THRESHOLD, 
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     MODEL_PATH)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory,get_kb_file_details
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from configs import logger, log_verbose
import pandas as pd
import re

def get_doc_name(knowledge_base_name,query):
    result_doc = ["未查询到相关文档","未查询到相关文档后缀"]
    doc_details = pd.DataFrame(get_kb_file_details(knowledge_base_name))
    all_file_name = list(doc_details.loc[:]["file_name"])
    file_names_list = []
    file_suffx_list = []
    for i in range(len(all_file_name)):
        if "/" not in all_file_name[i]:
            name = ""
            if len(all_file_name[i].split(".")) > 2:
                for j in range(len(all_file_name[i].split("."))-1):
                    name = name + all_file_name[i].split(".")[j] + "."
            file_names_list.append(name.strip("."))
            file_suffx_list.append(all_file_name[i].split(".")[-1])
    # 通过连续词的组合匹配书名
    for j in range(len(file_names_list)):
        start = 0
        i = 2
        low = start
        high = low+i # 最高位取不到，所以加2，匹配字数最少两个字
        while high<=len(file_names_list[j]):
            label = False
            while high<=len(file_names_list[j]):
                if file_names_list[j][low:high] in query:
                    result_doc[0] = file_names_list[j]
                    result_doc[1] = file_suffx_list[j]
                    label = True
                    break
                low += 1
                high += 1
            if label:
                break
            start += 1
            i+=1
            low = start
            high = low+i
    return result_doc[0],result_doc[1]


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              file_names: str=Body(..., description="文件名称(None表示搜库)", examples=["filename1, filename2"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]


    # 标记是否锁定文档
    if "只搜索" in query or "只与" in query or "文件" in query or "文档" in query:
        mark = True
        #######提取书名和文件后缀的操作，后缀不带点########
        doc_name,doc_suffx = get_doc_name(knowledge_base_name,query)
    elif "全数据库搜索" in query:
        mark = False
        doc_name = None
        doc_suffx = None
    else:
        mark = False
        doc_name = None
        doc_suffx = None

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            history: Optional[List[History]],
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        docs,label = await run_in_threadpool(search_docs,
                                       mark=mark,
                                       file_names=file_names,
                                       file_name=doc_name,
                                       file_suffx=doc_suffx,
                                       query=query,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold,
                                       )
        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL,"BAAI/bge-reranker-large")
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=embedding_device(),
                                            max_length=RERANKER_MAX_LENGTH,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
        
        if label=="F":
            # 当为文档对话时启用该数据处理方式
            print(docs)
            context = docs[0].page_content.replace("\n","")
            for i in range(1,len(docs)):
                context = re.sub("(?P<name>[\w\W]{0,})(?P=name)", '\g<name>', context+docs[i].page_content.replace("\n",""))
        else:
            context = "\n".join([doc.page_content for doc in docs])
        
        if mark and label=="F":
            # 当触发文档对话且确实搜到相关文档时启动该prompt
            prompt_template = get_prompt_template("knowledge_base_chat", "file_dialogue")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

        # if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
        #     prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        # elif mark:
        #     prompt_template = get_prompt_template("knowledge_base_chat", "file_dialogue")
        # else:
        #     prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            if label == "NF_NM":
                yield json.dumps({"answer": "根据你的query并没有在数据库中找到相关文档且未匹配到任何相关数据，以下显示的相关数据为全数据库扫描所得，与你的query并不相关，因此回复也可能并不相关，回复："+answer,
                                "docs": source_documents},
                                ensure_ascii=False)
            else:
                yield json.dumps({"answer": answer,
                                "docs": source_documents},
                                ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, history,model_name,prompt_name))

