import numpy as np
import pandas as pd
from intent_classifier import classify_intent
from entity_extractor import extract_entities
from entity_extractor import get_cypher_params
from graph_retrieval import GraphRetriever
from db_manager import DBManager
#from embeddings import DualEmbeddingRetriever #search_by_query

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
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import numpy as np
import os
import time
from transformers import AutoTokenizer

import pandas as pd
load_dotenv()

class LangChainWrapper(LLM):
    client: Any = Field(...)
    max_tokens: int = 500  # sets a default max output length

    def _llm_type(self) -> str:
        return "mistral_hf_api"

    # what LangChain calls when it needs the LLM to answer something
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.chat_completion(  # call the HuggingFace API
            messages=[{"role": "user", "content": prompt}],
            # Wrap the plain text prompt into chat format because Gemma ONLY understands chat messages.
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message["content"]


#################################### 1) BASELINE ###################################################################

def process_query_for_baseline(user_query,driver):
    # Step 1: Preprocessing
    intent_confidence = classify_intent(user_query)
    intent = intent_confidence['intent']
    print(intent)

    entities = extract_entities(user_query)
    print(entities)
    params = get_cypher_params(entities)
    print(params)


    graph_retrieval = GraphRetriever(driver)

    context = []
    # hotel_name,hotel_name_2, city, country, traveller_type, gender, age_group, rating, cleanliness_base, comfort_base, facilities_base

    if intent == "hotel_recommendation": #Query: Recommend me a good hotel in Tokyo (Japan)
        context.append(graph_retrieval.get_top_rated_hotels_in_city(params['city']))
        context.append(graph_retrieval.get_top_rated_hotels_in_country(params['country']))
    elif intent == "hotel_search": # Query: Find hotels in Paris
        context.append(graph_retrieval.get_hotels_by_rating())
        context.append(graph_retrieval.get_hotels_in_city_by_rating(params['city']))
        context.append(graph_retrieval.get_hotels_in_country_by_rating(params['country']))
    elif intent == "hotel_info": # Query: Tell me about The Azure Tower
        context.append(graph_retrieval.get_hotel_info(params['hotel_name']))
    elif intent == "review_query": # Query: Show me reviews for The Azure Tower
        context.append(graph_retrieval.get_hotel_reviews(params['hotel_name']))
        context.append(graph_retrieval.get_hotel_review_count(params['hotel_name']))
        context.append(graph_retrieval.get_latest_hotel_reviews(params['hotel_name']))
        context.append(graph_retrieval.get_hotels_with_most_reviews())
        context.append(graph_retrieval.get_hotel_reviews_by_travelers_from_country(params['hotel_name'], params['country']))
        context.append(graph_retrieval.get_hotels_reviewed_by_travellers_from_country(params['country']))
        context.append(graph_retrieval.get_hotels_reviewed_by_travellers_from_city(params['city']))
        context.append(graph_retrieval.get_hotel_reviews_filtered(params['hotel_name'], params.get('gender'),params.get('age_group'),params.get('traveller_type')))
    elif intent == "comparison": # Query: Compare the following hotels: The Azure Tower and L'Étoile Palace
        context.append(graph_retrieval.compare_two_hotels(params['hotel_name'], params['hotel_name_2']))
    elif intent == "traveller_preference": # Query: Best hotels for business travelers
        context.append(graph_retrieval.get_best_hotels_for_traveller_type(params['traveller_type']))
        context.append(graph_retrieval.get_best_hotels_for_gender(params['gender']))
        context.append(graph_retrieval.get_best_hotels_for_age_group(params['age_group']))
    elif intent == "location_query": # Query: Where is The Golden Oasis located?
        context.append(graph_retrieval.get_hotels_in_city(params['city']))
        context.append(graph_retrieval.get_hotels_in_country(params['country']))
        context.append(graph_retrieval.get_cities_in_country(params['country']))
        context.append(graph_retrieval.get_cities_With_Hotels())
        context.append(graph_retrieval.get_countries_With_Hotels())
    elif intent == "visa_query": #Query: Do I need a visa to travel from India to Dubai?
        context.append(graph_retrieval.get_countries_requiring_visa(params['country']))
        context.append(graph_retrieval.get_countries_not_requiring_visa(params['country']))
        context.append(graph_retrieval.get_hotels_accessible_without_visa(params['country']))
        context.append(graph_retrieval.get_hotels_with_no_visa_requirements())
    elif intent == "rating_filter": # Query: Hotels with cleanliness rating above 9
        context.append(graph_retrieval.get_average_hotel_rating_by_travellers_from_country(params['hotel_name'],params['country']))
        context.append(graph_retrieval.get_hotels_by_cleanliness_base(params['cleanliness_base']))
        context.append(graph_retrieval.get_hotels_by_comfort_base(params['comfort_base']))
        context.append(graph_retrieval.get_hotels_by_facilities_base(params['facilities_base']))
    elif intent == "general_question": # Query: How many hotels do you have?
        context.append(graph_retrieval.get_all_hotels())

    return context


#################################### 2) EMBEDDING ###################################################################
def build_shared_retriever(db_manager: Optional[DBManager] = None):
    # Try to construct a reusable retriever from your project's DualEmbeddingRetriever
    try:
        pdf_path = "./documents/ms1_checklist.pdf"

        reader = PdfReader(pdf_path)
        all_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

        # chunk using langchain
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        )
        documents = splitter.create_documents([all_text])
        len(documents)

        # embedding using langchain
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # create FAISS retriever
        vectorstore = LCFAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        # If you have DualEmbeddingRetriever implemented, uncomment and use it:
        # retriever = DualEmbeddingRetriever(db_manager.driver)
        # return retriever
        return retriever  # force fallback in this template
    except Exception:
        # Fallback: return a dummy retriever so code paths work without the full embeddings pipeline
        return "FAILED_TO_BUILD_RETRIEVER"


def create_retrieval_qa_for_model(model, retriever, HF_TOKEN):
    client = InferenceClient(
        model=model,
        token=HF_TOKEN
    )


    # Instantiate the wrapper
    llm = LangChainWrapper(client=client)


    # build rag qa using langchain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # concatenate all retrieved documents and feed them to the LLM as one big prompt.
    )
    return qa_chain

# Specific factories that match your skeletons and return a QA chain (reusing retriever)
def create_Gemma_llm(retriever=None, HF_TOKEN=None):
    model="google/gemma-2-2b-it"
    return create_retrieval_qa_for_model(model, retriever, HF_TOKEN)

def create_Mistral_llm(retriever=None, HF_TOKEN=None):
    model="mistralai/Mistral-7B-Instruct-v0.2"
    return create_retrieval_qa_for_model(model, retriever,HF_TOKEN)

def create_LLama_llm(retriever=None, HF_TOKEN=None):
    model="meta-llama/Llama-3.1-8B-Instruct"
    return create_retrieval_qa_for_model(model, retriever, HF_TOKEN)


#################################### 3) TESTING ###################################################################

def semantic_similarity(semantic_model_name,answer: str, reference: str) -> float:
    semantic_model = SentenceTransformer(semantic_model_name)

    emb = semantic_model.encode(
        [answer, reference],
        normalize_embeddings=True
    )
    # cosine similarity because vectors are normalized
    return float(np.dot(emb[0], emb[1]))


def evaluate_model(qa_chain,model_name, tokenizer_name,question,model_pricing,reference_answer=None,cost_per_1k_tokens=0.0):


    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2") #loads the exact tokenizer used by the model so you can convert text into tokens, the same way the model does internally.

    pricing = model_pricing[tokenizer_name]

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
        semantic_score = semantic_similarity("sentence-transformers/all-MiniLM-L6-v2",response, reference_answer)

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


#################################### 4) MAIN FUNCTION ###################################################################

def call_llm(query):

    db_manager = DBManager()
    HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
    if not HF_TOKEN:
        print("HUGGING_FACE_API_KEY not found in environment variables.")


    #  -------------- 1) get baseline --------------


    # context = process_query_for_baseline(query, db_manager.driver)
    # print(context)
    #
    # Persona = "You are a knowledgeable and friendly hotel recommender assistant and your name is Jarvis."
    # Task = f'''
    #         {Persona} Start any reply with Sir.
    #
    #         Your goal is given the user's query, help users choose hotels that best match their intents like
    #         location preferences, comfort needs and so on.
    #
    #         Compare multiple hotel options objectively, highlight trade-offs, and provide concise, practical recommendations.
    #
    #         You avoid exaggeration, do not invent hotel details, and prioritize user preferences over generic popularity.
    #
    #         User's query: {query}
    #
    #         Use the following data (that was retrieved based on the query) as context/baseline information to help with the recommendations: {context}
    #         '''

    #  -------------- 2) Build or reuse a shared retriever (embeddings + vectorstore). --------------
    shared_retriever = build_shared_retriever(db_manager)

    #  -------------- 3) Create a QA chain per model (these reuse shared_retriever) --------------
    gemma_qa = create_Gemma_llm(shared_retriever, HF_TOKEN)
    mistral_qa = create_Mistral_llm(shared_retriever, HF_TOKEN)
    llama_qa = create_LLama_llm(shared_retriever, HF_TOKEN)


    answer_gemma = gemma_qa.run(query)
    print("Gemma answer:\n", answer_gemma)

    answer_mistral = mistral_qa.run(query)
    print("Mistral answer:\n", answer_mistral)

    answer_llama = llama_qa.run(query)
    print("Llama answer:\n", answer_llama)

    # -------------- 4) Get accuracy or similarity if reference answer is available -------------------
    MODEL_PRICING = {
        "google/gemma-2-2b-it": {
            "input": 0.0001,  # per 1K tokens
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


    REFERENCE_ANSWER = """
    Milestone 1 report should include problem definition, dataset description,
    related work, methodology, and preliminary results.
    """

    comparsion_gemma = evaluate_model(
        gemma_qa,
        model_name="Gemma-2-2B",
        tokenizer_name="google/gemma-2-2b-it",
        question=query,
        model_pricing= MODEL_PRICING,
        reference_answer=REFERENCE_ANSWER
    )

    comparison_Mistral = evaluate_model(
        mistral_qa,
        model_name="Mistral-7B",
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
        question=query,
        model_pricing=MODEL_PRICING,
        reference_answer=REFERENCE_ANSWER
    )

    comparison_llama = evaluate_model(
        llama_qa,
        model_name="LLaMA-3.1-8B",
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        question=query,
        model_pricing=MODEL_PRICING,
        reference_answer=REFERENCE_ANSWER
    )

    #  -------------- RETURNS  --------------
    results = []
    results.append(comparsion_gemma)
    results.append(comparison_Mistral)
    results.append(comparison_llama)

    df = pd.DataFrame(results)
    print(df[["model","latency_sec","input_tokens","output_tokens","total_cost_usd","semantic_accuracy"]])
    df.to_csv("chatbot_comparison.csv", index=False)

    return answer_gemma,answer_mistral, answer_llama


if __name__ == "__main__":

    query = "What should be included in report Milestone 1?"
    answer_gemma,answer_mistral, answer_llama = call_llm(query)
