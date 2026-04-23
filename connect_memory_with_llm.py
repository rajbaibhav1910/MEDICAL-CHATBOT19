import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACEHUB_API_TOKEN = HF_TOKEN  # alias expected by some clients
if HUGGINGFACEHUB_API_TOKEN:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

huggingface_repo_id = os.environ.get("HF_REPO_ID", "HuggingFaceH4/zephyr-7b-beta")


def load_llm(repo_id: str):
    if not HUGGINGFACEHUB_API_TOKEN:
        raise RuntimeError(
            "Missing HF token. Set environment variable HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        temperature=0.5,
        max_new_tokens=512,
        top_p=0.9,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    chat_llm = ChatHuggingFace(llm=endpoint)
    return chat_llm

DB_FAISS_PATH = "vectorstore/db_faiss"
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# Now invoke with a single query
default_query = os.environ.get("USER_QUERY", "What are common symptoms of diabetes?")
user_query = input("Write Query Here: ") or default_query
response = qa_chain.invoke({"query": user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])