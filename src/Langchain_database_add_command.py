

import os
import re
import uuid
from langchain.docstore.document import Document  # 使用内置的 Document 类
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
import config_path

# class Document:
#     def __init__(self, page_content, metadata=None):
#         self.page_content = page_content
#         self.metadata = metadata

#     def __repr__(self):
#         return f"Document(page_content={self.page_content[:30]}..., metadata={self.metadata})"

help_file_path = f'{config_path.Database_PATH}/openfoam_command_helps.txt'

loader = TextLoader(help_file_path)
pages = loader.load()


pattern = re.compile(r"```input_file_begin:(.*?)input_file_end.```", re.DOTALL)
matches = pattern.findall(pages[0].page_content)
documents = [
    Document(
        page_content=match.strip(),
        metadata={'source': help_file_path, 'id': str(uuid.uuid4())}
    )
    for match in matches
]

persist_directory = f'{config_path.Database_PATH}/openfoam_command_helps'

batch_size = config_path.batchsize

for i in range(0, len(documents), batch_size):
    print("i:",i)
    if(i+batch_size<=len(documents)-1):
        batch = documents[i:i + batch_size]
    elif(i<=len(documents)-2):
        batch = documents[i:]

    if(i==0):
        vectordb = FAISS.from_documents(
            documents=batch, 
            embedding=HuggingFaceEmbeddings())
    else:
        vectordb.add_documents(documents=batch)


vectordb.save_local(persist_directory)




