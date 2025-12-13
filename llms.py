from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace
from langchain.llms.base import LLM
from typing import Optional, List, Any
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
import numpy as np

import time
from transformers import AutoTokenizer
from difflib import SequenceMatcher
import pandas as pd




# 1) prepare vectordb
#Load your PDF
pdf_path = "./ms1_checklist.pdf"

reader = PdfReader(pdf_path)
all_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text += text + "\n"

# Preview the start of the text
print("Extracted text preview:\n")
print(all_text[:500])


# 2) model
print('-' * 400)
########################### GEMMA ######################################################
from huggingface_hub import InferenceClient

HF_TOKEN = "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"


client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=HF_TOKEN
)


# 3) RAG using langchain
# Custom LLM wrapper for HuggingFace Inference Client (Gemma conversational)
class GemmaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500 #sets a default max output length

    @property
    def _llm_type(self) -> str:
        return "gemma_hf_api" #Identify the LLM type

    #what LangChain calls when it needs the LLM to answer something
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion( #call the HuggingFace API
            messages=[{"role": "user", "content": prompt}],  #Wrap the plain text prompt into chat format because Gemma ONLY understands chat messages.
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]


# Instantiate the wrapper
gemma_llm = GemmaLangChainWrapper(client=client)


# chunk using langchain
splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
documents1 = splitter1.create_documents([all_text])
len(documents1)


# embedding using langchain
embedding_model1 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create FAISS retriever
vectorstore1 = LCFAISS.from_documents(
    documents=documents1,
    embedding=embedding_model1
)
retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 10})


# build rag qa using langchain
qa_chain1 = RetrievalQA.from_chain_type(
    llm=gemma_llm,
    retriever=retriever1,
    chain_type="stuff" #concatenate all retrieved documents and feed them to the LLM as one big prompt.
)

response1 = qa_chain1.run("What should be included in report Milestone 1?")
print(response1)


##################################### Mistral##############################################
print('-' * 400)
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

class MistralLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "mistral_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]


# Instantiate the wrapper
mistral_llm = MistralLangChainWrapper(client=client)


# chunk using langchain
splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
documents2 = splitter2.create_documents([all_text])
len(documents2)


# embedding using langchain
embedding_model2 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create FAISS retriever
vectorstore2 = LCFAISS.from_documents(
    documents=documents2,
    embedding=embedding_model2
)
retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 10})


# build rag qa using langchain
qa_chain2 = RetrievalQA.from_chain_type(
    llm=mistral_llm,
    retriever=retriever2,
    chain_type="stuff" #concatenate all retrieved documents and feed them to the LLM as one big prompt.
)

response2 = qa_chain2.run("What should be included in report Milestone 1?")
print(response2)

################################LLama###################################################
print('-' * 400)
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

class LlamaLangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "llama_hf_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]

# Instantiate the wrapper
llama_llm = LlamaLangChainWrapper(client=client)


# chunk using langchain
splitter3 = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
documents3 = splitter3.create_documents([all_text])
len(documents3)


# embedding using langchain
embedding_model3 = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# create FAISS retriever
vectorstore3 = LCFAISS.from_documents(
    documents=documents3,
    embedding=embedding_model3
)
retriever3 = vectorstore3.as_retriever(search_kwargs={"k": 10})


# build rag qa using langchain
qa_chain3 = RetrievalQA.from_chain_type(
    llm=llama_llm,
    retriever=retriever3,
    chain_type="stuff" #concatenate all retrieved documents and feed them to the LLM as one big prompt.
)

response3 = qa_chain3.run("What should be included in report Milestone 1?")
print(response3)


#-----------------------------------------------------------------------------------------
# HuggingFace API pricing (as of 2024, approximate)
MODEL_PRICING = {
    "google/gemma-2-2b-it": {
        "input": 0.0001,   # per 1K tokens
        "output": 0.0002
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "input": 0.0002,
        "output": 0.0006
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "input": 0.0003,
        "output": 0.0008
    }
}

semantic_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def semantic_similarity(answer: str, reference: str) -> float:
    emb = semantic_model.encode(
        [answer, reference],
        normalize_embeddings=True
    )
    # cosine similarity because vectors are normalized
    return float(np.dot(emb[0], emb[1]))


def evaluate_model(qa_chain,model_name, tokenizer_name,question,reference_answer=None,cost_per_1k_tokens=0.0):

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") #loads the exact tokenizer used by the model so you can convert text into tokens, the same way the model does internally.

    pricing = MODEL_PRICING[tokenizer_name]

    # ---- timing ----
    start = time.time()
    response = qa_chain.run(question)
    latency = time.time() - start

    # ---- token estimation ----
    input_tokens = len(tokenizer.encode(question))
    output_tokens = len(tokenizer.encode(response))

    # ---- cost calculation ----
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    # ---- accuracy (optional) ----
    semantic_score = None
    if reference_answer:
        semantic_score = semantic_similarity(response, reference_answer)

    return {
        "model": model_name,
        "latency_sec": round(latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "semantic_accuracy": round(semantic_score, 3),
        "response": response
    }

REFERENCE_ANSWER = """
Milestone 1 report should include problem definition, dataset description,
related work, methodology, and preliminary results.
"""

question = "What should be included in report Milestone 1?"



results = []

results.append(evaluate_model(
    qa_chain1,
    model_name="Gemma-2-2B",
    tokenizer_name="google/gemma-2-2b-it",
    question=question,
    reference_answer=REFERENCE_ANSWER
))

results.append(evaluate_model(
    qa_chain2,
    model_name="Mistral-7B",
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
    question=question,
    reference_answer=REFERENCE_ANSWER
))

results.append(evaluate_model(
    qa_chain3,
    model_name="LLaMA-3.1-8B",
    tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
    question=question,
    reference_answer=REFERENCE_ANSWER
))


df = pd.DataFrame(results)
print(df[[
    "model",
    "latency_sec",
    "input_tokens",
    "output_tokens",
    "total_cost_usd",
    "semantic_accuracy"
]])

df.to_csv("comparison.csv", index=False)