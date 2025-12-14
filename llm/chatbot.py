import numpy as np
import pandas as pd
from intent_classifier import classify_intent
from entity_extractor import extract_entities
from entity_extractor import get_cypher_params
import graph_retrieval

def process_query(self, user_query):
    # Step 1: Preprocessing
    intent = classify_intent(user_query)

    entities = extract_entities(user_query)
    params = get_cypher_params(entities)

    context = []

    if intent == "hotel_recommendation": #Query: Recommend me a good hotel in Tokyo
        context.append(graph_retrieval.get_top_rated_hotels_in_city(params['city']))
        context.append(graph_retrieval.get_top_rated_hotels_in_country(params['city']))
    elif intent == "hotel_search": # Query: Find hotels in Paris
        context.append(graph_retrieval.get_hotels_by_rating(params['rating']))
        context.append(graph_retrieval.get_hotels_in_city_by_rating(params['city'], params['rating']))
        context.append(graph_retrieval.get_hotels_in_country_by_rating(params['country'], params['rating']))
        context.append(graph_retrieval.get_hotels_by_cleanliness_base(params['cleanliness_base']))
        context.append(graph_retrieval.get_hotels_by_comfort_base(params['comfort_base']))
        context.append(graph_retrieval.get_hotels_by_facilities_base(params['facilities_base']))
    elif intent == "hotel_info": # Query: Tell me about The Royal Compass
        context.append(graph_retrieval.get_hotel_rating(params['hotel_name']))
    elif intent == "review_query": # Query: Show me reviews for this hotel
        context.append(graph_retrieval.get_hotel_reviews(params['hotel_name']))
        context.append(graph_retrieval.get_hotel_review_count(params['hotel_name']))
        context.append(graph_retrieval.get_latest_hotel_reviews(params['hotel_name']))
        context.append(graph_retrieval.get_hotels_with_most_reviews())
        context.append(graph_retrieval.get_hotel_reviews_by_travelers_from_country(params['hotel_name'], params['country']))
        context.append(graph_retrieval.get_hotels_reviewed_by_travellers_from_country(params['country']))
        context.append(graph_retrieval.get_hotels_reviewed_by_travellers_from_city(params['city']))
        context.append(graph_retrieval.get_hotel_reviews_filtered(params['hotel_name'], params.get('gender'),params.get('age_group'),params.get('traveller_type')))
    elif intent == "comparison": # Query: Compare The Azure Tower and Marina Bay
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
        # hotel_name,hotel_name_2, city, country,from_country,to_country, traveller_type, gender, age_group, rating, cleanliness_base, comfort_base, facilities_base

        baseline_queries = ["get_countries_requiring_visa(from_country)", "get_countries_not_requiring_visa(from_country)", "get_hotels_accessible_without_visa(from_country)", "get_hotels_with_no_visa_requirements(self)"]
        context.append(graph_retrieval.get_countries_requiring_visa(params['from_country']))
        context.append(graph_retrieval.get_countries_not_requiring_visa(params['from_country']))
        context.append(graph_retrieval.get_hotels_accessible_without_visa(params['from_country']))
        context.append(graph_retrieval.get_hotels_with_no_visa_requirements())
    elif intent == "rating_filter": # Query: Hotels with cleanliness rating above 9
        baseline_queries = ["get_average_hotel_rating_by_travellers_from_country(hotel_name, country)"]
        context.append(graph_retrieval.get_average_hotel_rating_by_travellers_from_country(params['hotel_name'],
                                                                                           params['country']))
    elif intent == "general_question": # Query: How many hotels do you have?
        baseline_queries = ""

    # INTENTS = [
    #     "hotel_recommendation",  # 0
    #     "hotel_search",  # 1
    #     "hotel_info",  # 2
    #     "review_query",  # 3
    #     "comparison",  # 4
    #     "traveller_preference",  # 5
    #     "location_query",  # 6
    #     "visa_query",  # 7
    #     "rating_filter",  # 8
    #     "general_question"  # 9
    # ]