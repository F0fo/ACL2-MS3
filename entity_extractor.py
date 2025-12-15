import spacy
import pandas as pd

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
        "demographics": []
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


    return params


if __name__ == "__main__":
    test_queries = [
        "Find hotels in Paris",
        "Best hotels for business travelers",
        "Do I need a visa from India to Japan?",
        "Show me reviews for The Azure Tower",
        "Best hotels for solo female travelers",
        "Hotels for travelers aged 25-34",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        entities = extract_entities(query)
        params = get_cypher_params(entities)
        print(f"  Entities: {entities}")
        print(f"  Params: {params}")
