import spacy
import pandas as pd
import re

# run in terminal: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Load domain data
hotels_df = pd.read_csv("data/hotels.csv")
visa_df = pd.read_csv("data/visa.csv")

HOTELS = hotels_df["hotel_name"].tolist()
CITIES = hotels_df["city"].unique().tolist()
COUNTRIES = list(set(hotels_df["country"].tolist() + visa_df["from"].tolist() + visa_df["to"].tolist()))

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def extract_entities(text):
    """Extract entities from user query using spaCy NER."""
    doc = nlp(text)

    entities = {
        "hotels": [],
        "cities": [],
        "countries": [],
        "traveller_types": [],
        "demographics": [],
        "cleanliness_base": None,
        "comfort_base": None,
        "facilities_base": None
    }

    # Use spaCy NER to extract entities
    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_lower = ent_text.lower()

        # Hotels - FAC, EVENT, WORK_OF_ART, ORG labels
        if ent.label_ in ["FAC", "EVENT", "WORK_OF_ART", "ORG"]:
            for hotel in HOTELS:
                if hotel.lower() == ent_lower or hotel.lower() in ent_lower:
                    if hotel not in entities["hotels"]:
                        entities["hotels"].append(hotel)

        # Cities/Countries - GPE label
        if ent.label_ == "GPE":
            for city in CITIES:
                if city.lower() == ent_lower:
                    if city not in entities["cities"]:
                        entities["cities"].append(city)
            for country in COUNTRIES:
                if country.lower() == ent_lower:
                    if country not in entities["countries"]:
                        entities["countries"].append(country)

        # Age groups - DATE label
        if ent.label_ == "DATE":
            if ent_text in AGE_GROUPS:
                if ent_text not in entities["demographics"]:
                    entities["demographics"].append(ent_text)

    # Traveller types and gender - use tokens (no NER label exists)
    for token in doc:
        token_lower = token.text.lower()

        # Traveller types
        if token_lower in ["solo", "alone"]:
            if "Solo" not in entities["traveller_types"]:
                entities["traveller_types"].append("Solo")
        elif token_lower in ["family", "families"]:
            if "Family" not in entities["traveller_types"]:
                entities["traveller_types"].append("Family")
        elif token_lower in ["couple", "couples"]:
            if "Couple" not in entities["traveller_types"]:
                entities["traveller_types"].append("Couple")
        elif token_lower in ["business", "corporate"]:
            if "Business" not in entities["traveller_types"]:
                entities["traveller_types"].append("Business")

        # Demographics - gender (no NER label exists)
        if token_lower in ["female", "women", "woman"]:
            if "Female" not in entities["demographics"]:
                entities["demographics"].append("Female")
        elif token_lower in ["male", "men", "man"]:
            if "Male" not in entities["demographics"]:
                entities["demographics"].append("Male")

        # Demographics - age groups (no NER label exists)
        if token_lower in ["seniors", "senior", "elderly", "older", "retirees", "retired"]:
            if "55+" not in entities["demographics"]:
                entities["demographics"].append("55+")
        elif token_lower in ["young", "youth", "younger"]:
            if "18-24" not in entities["demographics"]:
                entities["demographics"].append("18-24")
        elif token_lower in ["millennials", "millennial"]:
            if "25-34" not in entities["demographics"]:
                entities["demographics"].append("25-34")

    # Fallback: Direct text matching for hotel names (in case spaCy NER misses them)
    text_lower = text.lower()
    if not entities["hotels"]:
        for hotel in HOTELS:
            hotel_lower = hotel.lower()
            # Check if hotel name appears in text (handle partial matches like "azure tower" for "The Azure Tower")
            # Remove common prefixes for matching
            hotel_core = hotel_lower.replace("the ", "").replace("l'", "").replace("la ", "")
            if hotel_lower in text_lower or hotel_core in text_lower:
                if hotel not in entities["hotels"]:
                    entities["hotels"].append(hotel)

    # Extract rating criteria (cleanliness, comfort, facilities)

    # Keywords for each category
    cleanliness_keywords = ["cleanliness", "clean", "hygiene", "hygienic", "tidy", "spotless"]
    comfort_keywords = ["comfort", "comfortable", "cozy", "cosy", "relaxing"]
    facilities_keywords = ["facilities", "amenities", "services", "features", "equipment"]

    # Extract numeric rating values from text
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    rating_value = None
    for num in numbers:
        val = float(num)
        if 1 <= val <= 10:  # Rating scale is typically 1-10
            rating_value = val
            break

    # Check for each category and assign rating
    for keyword in cleanliness_keywords:
        if keyword in text_lower:
            entities["cleanliness_base"] = rating_value if rating_value else 7.0  # Default threshold
            break

    for keyword in comfort_keywords:
        if keyword in text_lower:
            entities["comfort_base"] = rating_value if rating_value else 7.0
            break

    for keyword in facilities_keywords:
        if keyword in text_lower:
            entities["facilities_base"] = rating_value if rating_value else 7.0
            break

    return entities


def get_cypher_params(entities):
    """Convert extracted entities to Cypher query parameters."""
    params = {}

    if entities["hotels"]:
        params["hotel_name"] = entities["hotels"][0]
        if len(entities["hotels"]) > 1:
            params["hotel_name_2"] = entities["hotels"][1]
        else:
            params["hotel_name_2"] = ""
    else:
        params["hotel_name"] = ""
        params["hotel_name_2"] = ""

    if entities["cities"]:
        params["city"] = entities["cities"][0]
    else:
        params["city"] = ""

    if entities["countries"]:
        params["country"] = entities["countries"][0]
    else:
        params["country"] = ""


    if entities["traveller_types"]:
        params["traveller_type"] = entities["traveller_types"][0]
    else:
        params["traveller_type"] = ""


    if entities["demographics"]:
        for d in entities["demographics"]:
            if d in ["Male", "Female"]:
                params["gender"] = d
                params["age_group"] = ""
            else:
                params["age_group"] = d
                params["gender"] = ""
    else:
        params["gender"] = ""
        params["age_group"] = ""

    # Rating criteria parameters
    params["cleanliness_base"] = entities.get("cleanliness_base")
    params["comfort_base"] = entities.get("comfort_base")
    params["facilities_base"] = entities.get("facilities_base")

    return params


if __name__ == "__main__":
    test_queries = [
        "Find hotels in Paris",
        "Best hotels for business travelers",
        "Do I need a visa from India to Japan?",
        "Show me reviews for The Azure Tower",
        "Best hotels for solo female travelers",
        "Hotels for travelers aged 25-34",
        "Hotels with cleanliness rating above 9",
        "Find comfortable hotels in Tokyo",
        "Hotels with good facilities and amenities",
        "Clean hotels with comfort rating 8",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        entities = extract_entities(query)
        params = get_cypher_params(entities)
        print(f"  Entities: {entities}")
        print(f"  Params: {params}")
