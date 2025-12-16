import numpy as np
import pandas as pd
from intent_classifier import classify_intent
from entity_extractor import extract_entities
from entity_extractor import get_cypher_params
from graph_retrieval import GraphRetriever
from db_manager import DBManager
from embeddings_retreiver import HotelEmbeddingRetriever  # Unified retriever with location filtering

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
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

    @property
    def _llm_type(self) -> str:
        return "mistral_hf_api"

    # what LangChain calls when it needs the LLM to answer something
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            # Truncate prompt if too long
            max_prompt_chars = 3000
            if len(prompt) > max_prompt_chars:
                prompt = prompt[:max_prompt_chars] + "\n\n[Context truncated for length]"

            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message["content"]
        except Exception as e:
            error_msg = str(e)
            print(f"LLM API Error: {error_msg}")
            return f"Error generating response: {error_msg[:200]}"


class DualEmbeddingRetrieverWrapper(BaseRetriever):
    """Wrapper to make DualEmbeddingRetriever compatible with LangChain"""
    retriever: Any = Field(...)
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve documents using HotelEmbeddingRetriever"""
        try:
            # Use the search_by_query method from our custom retriever
            results = self.retriever.search_by_query(query, top_k=self.k)

            # Convert results to LangChain Documents
            documents = []
            for result in results:
                # Create document content from hotel info
                # Note: results use 'hotel' key for name, not 'name'
                hotel_name = result.get('hotel') or result.get('name', 'Unknown')
                content = f"Hotel: {hotel_name}\n"
                content += f"City: {result.get('city', 'Unknown')}, Country: {result.get('country', 'Unknown')}\n"
                content += f"Star Rating: {result.get('star_rating', 'N/A')}\n"
                content += f"Average Score: {result.get('avg_score', 'N/A')}\n"
                content += f"Cleanliness: {result.get('cleanliness', 'N/A')}\n"
                content += f"Comfort: {result.get('comfort', 'N/A')}\n"
                content += f"Facilities: {result.get('facilities', 'N/A')}\n"
                content += f"Location Score: {result.get('location', 'N/A')}\n"

                doc = Document(
                    page_content=content,
                    metadata={"hotel_id": result.get("hotel_id"), "score": result.get("score")}
                )
                documents.append(doc)

            return documents
        except Exception as e:
            print(f"Error in retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []


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
    # Try to construct a reusable retriever from HotelEmbeddingRetriever (with location filtering)
    try:
        # Create the unified hotel retriever (uses 'feature' model by default)
        hotel_retriever = HotelEmbeddingRetriever(db_manager.driver, model_type='feature')

        # Wrap it for LangChain compatibility
        retriever = DualEmbeddingRetrieverWrapper(retriever=hotel_retriever, k=5)

        return retriever
    except Exception as e:
        print(f"Failed to build retriever: {e}")
        import traceback
        traceback.print_exc()
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

def format_context_for_llm(context):
    """Format the baseline context into a readable string for the LLM"""
    if not context:
        return "No context available."

    formatted = []
    total_chars = 0
    MAX_CHARS = 4000  # Increased limit for more context

    # Important property keys to include
    IMPORTANT_KEYS = [
        'name', 'text', 'city', 'country', 'star_rating',
        'score_overall', 'score_cleanliness', 'score_comfort', 'score_facilities', 'score_location',
        'avg_cleanliness', 'avg_comfort', 'avg_facilities', 'avg_location', 'avg_reviewer_score',
        'review_count', 'type', 'gender', 'age_group'
    ]

    for idx, ctx in enumerate(context, 1):
        if not ctx or total_chars > MAX_CHARS:
            continue

        if isinstance(ctx, dict):
            if 'rows' in ctx and ctx['rows']:
                # Format tabular data
                formatted.append(f"\n--- Data Set {idx} ---")
                for row in ctx['rows'][:8]:  # Limit to 8 rows
                    row_str = str(row)
                    if total_chars + len(row_str) < MAX_CHARS:
                        formatted.append(row_str)
                        total_chars += len(row_str)

            elif 'graph' in ctx:
                # Format graph data - extract nodes by type
                nodes = ctx['graph'].get('nodes', [])
                hotels = []
                reviews = []
                other = []

                for node in nodes:
                    labels = node.get('labels', [])
                    props = node.get('properties', {})
                    if not props:
                        continue

                    # Filter to important properties
                    filtered = {k: v for k, v in props.items() if k in IMPORTANT_KEYS and v is not None}

                    if 'Hotel' in labels:
                        hotels.append(filtered)
                    elif 'Review' in labels:
                        # Truncate review text if too long
                        if 'text' in filtered and len(str(filtered['text'])) > 200:
                            filtered['text'] = str(filtered['text'])[:200] + "..."
                        reviews.append(filtered)
                    else:
                        other.append(filtered)

                # Add hotels first
                if hotels:
                    formatted.append(f"\n--- Hotels ---")
                    for h in hotels[:3]:
                        h_str = str(h)
                        if total_chars + len(h_str) < MAX_CHARS:
                            formatted.append(h_str)
                            total_chars += len(h_str)

                # Add reviews
                if reviews:
                    formatted.append(f"\n--- Reviews ---")
                    for r in reviews[:5]:  # Limit reviews
                        r_str = str(r)
                        if total_chars + len(r_str) < MAX_CHARS:
                            formatted.append(r_str)
                            total_chars += len(r_str)

        elif isinstance(ctx, list):
            formatted.append(f"\n--- Data {idx} ---")
            for item in ctx[:5]:
                item_str = str(item)
                if total_chars + len(item_str) < MAX_CHARS:
                    formatted.append(item_str)
                    total_chars += len(item_str)

    return "\n".join(formatted) if formatted else "No relevant data found."


def call_llm(query, baseline_context=None):

    db_manager = DBManager()
    HF_TOKEN = os.getenv("HUGGING_FACE_API_KEY")
    if not HF_TOKEN:
        print("HUGGING_FACE_API_KEY not found in environment variables.")

    #  -------------- 1) Format baseline context --------------
    context_str = format_context_for_llm(baseline_context) if baseline_context else ""

    Persona = "You are a knowledgeable and friendly hotel recommender assistant named Jarvis."
    enhanced_query = f'''{Persona}

Your goal is to help users with hotel recommendations based on the provided data.
Compare options objectively and provide practical recommendations.
Do not invent hotel details - only use information from the context provided.

User's query: {query}

Retrieved hotel data:
{context_str}

Please answer the user's query based on the above data.'''

    #  -------------- 2) Build or reuse a shared retriever (embeddings + vectorstore). --------------
    shared_retriever = build_shared_retriever(db_manager)

    #  -------------- 3) Create a QA chain per model (these reuse shared_retriever) --------------
    gemma_qa = create_Gemma_llm(shared_retriever, HF_TOKEN)
    mistral_qa = create_Mistral_llm(shared_retriever, HF_TOKEN)
    llama_qa = create_LLama_llm(shared_retriever, HF_TOKEN)

    # Use enhanced query with context
    answer_gemma = gemma_qa.run(enhanced_query)
    print("Gemma answer:\n", answer_gemma)

    answer_mistral = mistral_qa.run(enhanced_query)
    print("Mistral answer:\n", answer_mistral)

    answer_llama = llama_qa.run(enhanced_query)
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
