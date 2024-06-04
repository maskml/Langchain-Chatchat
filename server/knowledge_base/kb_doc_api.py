import os
import urllib
from fastapi import File, Form, Body, Query, UploadFile
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     logger, log_verbose, )
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
from server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path,
                                         files2docs_in_thread, KnowledgeFile)
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse
from pydantic import Json
import json
from server.knowledge_base.kb_service.base import KBServiceFactory, KBService, get_kb_file_details
from server.db.repository.knowledge_file_repository import get_file_detail
from langchain.docstore.document import Document
from server.knowledge_base.model.kb_document_model import DocumentWithVSId
from typing import List, Dict
import pandas as pd
import random

def search_docs(
        mark: bool = Body("", description="是否锁死文档", examples=["True"]),
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=1),
        file_names: str = Body("", description="文件名称，支持 sql 通配符", examples=["filename1,filename2"]),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        file_suffx: str = Body("", description="文件后缀，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[DocumentWithVSId]:
    # mark暂未用到
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    # 当为文档对话时label为F(find)，没找到文档但是匹配到数据NF_M(notfind_match)，没找到文档且没匹配到数据NF_NM
    label = "NF_NM"
    if kb is not None:
        # 优先级：文档对话>数据库搜索对话=传入的文档
        if file_name and file_name != "未查询到相关文档":
            print("*************************")
            print("进行文档对话")
            print("*************************")
            # 有相关文档
            label = "F"
            data = kb.list_docs(file_name=file_name+"."+file_suffx, metadata={})
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
        elif not file_name or file_name == "未查询到相关文档":
            # 没找到相关文档，则进行数据库匹配
            docs = kb.search_docs(query, top_k, score_threshold)
            if len(docs) > 0:
                # 若匹配到相关数据
                label = "NF_M"
                if file_names is not None and file_names != "":
                    print("*************************")
                    print("未搜索到文档，但是匹配到数据，且有文档输入")
                    print("*************************")
                    # 若传入了文件则在搜到的数据中筛选出对应的内容
                    file_names = file_names.split(",")
                    data = []
                    for x in docs:
                        filename = x[0].metadata.get("source")
                        if filename in file_names == True:
                            data.append(DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")))
                else:
                    # 若没有传入文件则返回匹配到的内容
                    print("*************************")
                    print("未搜索到文档，但是匹配到数据，且没有文档输入")
                    print("*************************")
                    data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            else:
                data = []
                if file_names is not None and file_names != "":
                    # 若有传入的文件则直接搜索对应的文件内容
                    print("*************************")
                    print("未搜索到文档，没匹配到数据，且有文档输入")
                    print("*************************")
                    file_names = file_names.split(",")
                    max_len = 0
                    result = []
                    for i in range(len(file_names)):
                        file_data = kb.list_docs(file_name=file_names[i], metadata={})
                        result.append(file_data)
                        max_len = len(file_data) if len(file_data) > max_len else max_len
                    for index in range(max_len):
                        for item in result:
                                if index < len(item) :
                                    data.append(item[index])
                else:
                    # 若未匹配到相关数据，则对整个数据库中所有文件进行扫描
                    print("*************************")
                    print("未搜索到文档，没匹配到数据，且没有文档输入")
                    print("*************************")
                    doc_details = pd.DataFrame(get_kb_file_details(knowledge_base_name))
                    all_file_name = list(doc_details.loc[:]["file_name"])
                    if len(all_file_name) >= top_k:
                        for i in range(len(all_file_name)):
                            file_data = kb.list_docs(file_name=all_file_name[i], metadata={})
                            for d in file_data:
                                if "vector" in d.metadata:
                                    del d.metadata["vector"]
                            data.append(file_data[0])
                    else:
                        # for i in range(len(all_file_name)):
                        #     file_data = kb.list_docs(file_name=all_file_name[i], metadata={})
                        k = 0
                        while len(data)< top_k and k<top_k:
                            for i in range(len(all_file_name)):
                                file_data = kb.list_docs(file_name=all_file_name[i], metadata={})
                                for d in file_data:
                                    if "vector" in d.metadata:
                                        del d.metadata["vector"]
                                # 防止出现文档的切片数少于k
                                try:
                                    data.append(file_data[k])
                                except:
                                    continue
                            k+=1
    return data,label

def cto_search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=1),
        file_name: str = Body("", description="文件名称，支持 sql 通配符", examples=["filename1,filename2"]),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[DocumentWithVSId]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    logger.info("knowlege base")
    logger.info(knowledge_base_name)
    if kb is not None:
       logger.info(kb.list_files())
    else:
       logger.info("kb null")
    logger.info(file_name)
    logger.info(metadata)

    file_flag = False
    filename_dict ={}

    if file_name is not None and file_name != "":
        file_flag  = True
        filename_dict = {filename.strip():True for filename in file_name.split(",")}
    logger.info("filenames")
    logger.info(file_name)
    logger.info(filename_dict)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k*2, score_threshold)
            logger.info("docs")
            logger.info(len(docs))
            if len(docs) > 0:
                if file_flag == True:
                    data = filterDocbynames(docs, filename_dict)
                else:
                    data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            else:
                if file_flag == True:
                    data = getalldocs(filename_dict, kb)
                else:
                    all_filename_dict = {filename.strip():True for filename in kb.list_files()}
                    data = getalldocs(all_filename_dict, kb)
        elif file_flag == True:
            data = getalldocs(filename_dict, kb)

    if len(data) > top_k:
        return data[:top_k]
    else:
        return data

def filterDocbynames(docs:List[Document], filename_dict: dict)->List[DocumentWithVSId]:
    data = []
    for x in docs:
        filename = x[0].metadata.get("source")
        if filename_dict.get(filename) == True:
            data.append(DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")))

    return data
def getalldocs(filenamedict: dict, kb: KBService)-> List[DocumentWithVSId]:
    data = []
    templists=[]
    maxlen = 0
    for file_name, exists in filenamedict.items():
        tempdict = {}
        tempdata = kb.list_docs(file_name,tempdict)
        if len(tempdata) > maxlen:
            maxlen =  len(tempdata)
        templists.append(tempdata)

    for index in range(maxlen):
        for doclist in templists:
            if index < len(doclist) :
                data.append(doclist[index])
    return data

def oldsearch_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=1),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[DocumentWithVSId]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k*2, score_threshold)
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)

    return data


def update_docs_by_id(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        docs: Dict[str, Document] = Body(..., description="要更新的文档内容，形如：{id: Document, ...}")
) -> BaseResponse:
    '''
    按照文档 ID 更新文档内容
    '''
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=500, msg=f"指定的知识库 {knowledge_base_name} 不存在")
    if kb.update_doc_by_ids(docs=docs):
        return BaseResponse(msg=f"文档更新成功")
    else:
        return BaseResponse(msg=f"文档更新失败")


# def rename_file(
#         knowledge_base_name: str,
#         file_name:str,
#         new_file_name:str
# )-> BaseResponse:

def rename_file(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_name: str = Body("", description="文件名称"),
        new_file_name: str = Body("", description="文件新名称"),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库{knowledge_base_name}")
    # elif file_name not in list_files(knowledge_base_name):
    #     return BaseResponse(code=404, msg=f"未在知识库 {knowledge_base_name}中找到{file_name}文件")
    else:
        try:
            kb.rename_file(file_name,new_file_name)
        except Exception as e:
            msg = f"重命名文件出错： {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                        exc_info=e if log_verbose else None)
            return BaseResponse(code=500, msg=msg)
        return BaseResponse(code=200, msg=f"已重命名文件 {new_file_name}")


def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)



def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                # TODO: filesize 不同后的处理
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


# TODO: 等langchain.document_loaders支持内存文件的时候再开通
# def files2docs(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件"),
#                 save: bool = Form(True, description="是否将文件保存到知识库目录")):
#     def save_files(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     def files_to_docs(files):
#         for result in files2docs_in_thread(files):
#             yield json.dumps(result, ensure_ascii=False)


def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
        override: bool = Form(False, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        docs: Json = Form({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    API接口：上传文件，并/或向量化
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    file_names = list(docs.keys())
    logger.info("filenames:")
    logger.info(file_names)
    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})


def delete_docs(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
        delete_content: bool = Body(False),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} 文件删除失败，错误信息：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"文件删除完成", data={"failed_files": failed_files})


def update_info(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        kb_info: str = Body(..., description="知识库介绍", examples=["这是一个知识库"]),
):
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    kb.update_info(kb_info)

    return BaseResponse(code=200, msg=f"知识库介绍修改完成", data={"kb_info": kb_info})


def update_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_names: List[str] = Body(..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: Json = Body({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    更新知识库文档
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        logger.info("filename2:")
        logger.info(file_name)
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    # 从文件生成docs，并进行向量化。
    # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
    for status, result in files2docs_in_thread(kb_files,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               zh_title_enhance=zh_title_enhance):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})


def download_doc(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        preview: bool = Query(False, description="是：浏览器内预览；否：下载"),
):
    """
    下载知识库文档
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{kb_file.filename} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{kb_file.filename} 读取文件失败")


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
    """

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            if kb.exists():
                kb.clear_vs()
            kb.create_kb()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            for status, result in files2docs_in_thread(kb_files,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    kb.add_doc(kb_file, not_refresh_vs_cache=True)
                else:
                    kb_name, file_name, error = result
                    msg = f"添加文件‘{file_name}’到知识库‘{knowledge_base_name}’时出错：{error}。已跳过。"
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            if not not_refresh_vs_cache:
                kb.save_vector_store()

    return EventSourceResponse(output())
