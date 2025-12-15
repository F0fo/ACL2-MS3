import numpy as np
import pandas as pd
from intent_classifier import classify_intent
from entity_extractor import extract_entities
from entity_extractor import get_cypher_params
from graph_retrieval import GraphRetriever
from db_manager import DBManager
#from embeddings import DualEmbeddingRetriever #search_by_query



def process_query(user_query,driver):
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
        # context.append(graph_retrieval.get_hotel_review_count(params['hotel_name']))
        # context.append(graph_retrieval.get_latest_hotel_reviews(params['hotel_name']))
        # context.append(graph_retrieval.get_hotels_with_most_reviews())
        # context.append(graph_retrieval.get_hotel_reviews_by_travelers_from_country(params['hotel_name'], params['country']))
        # context.append(graph_retrieval.get_hotels_reviewed_by_travellers_from_country(params['country']))
        # context.append(graph_retrieval.get_hotels_reviewed_by_travellers_from_city(params['city']))
        # context.append(graph_retrieval.get_hotel_reviews_filtered(params['hotel_name'], params.get('gender'),params.get('age_group'),params.get('traveller_type')))
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

if __name__ == "__main__":
    db_manager = DBManager()


    query = "Show me reviews for The Azure Tower"
    context = process_query(query, db_manager.driver)
    print(context)

    Persona = "You are a knowledgeable and friendly hotel recommender assistant and your name is Jarvis."
    Task = f'''
        {Persona} Start any reply with Sir. 
        
        Your goal is given the user's query, help users choose hotels that best match their intents like 
        location preferences, comfort needs and so on. 
        
        Compare multiple hotel options objectively, highlight trade-offs, and provide concise, practical recommendations. 
        
        You avoid exaggeration, do not invent hotel details, and prioritize user preferences over generic popularity.
        
        User's query: {query}
        
        Use the following data (that was retrieved based on the query) as context/baseline information to help with the recommendations: {context}
        '''

    #retriever = DualEmbeddingRetriever(db_manager.driver)
