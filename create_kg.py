#IMPORTS

####### UNCOMMENT IN THE END
from neo4j import GraphDatabase
import pandas as pd

# Connection to db and Set up #-------------------------------------------------------------

#read config.txt for Neo4j db details
def readConfig(filename = 'config.txt'):
    config = {}
    with open("config.txt") as c:
        for line in c:
            if '=' in line:
                key, value = line.strip().split("=") #for every line, everything before =  is key, everything after is that key's value
                config[key] = value
    print(f"Config loaded with \n URI : {config['URI']} \n Username: {config['USERNAME']}\n Password: {config['PASSWORD']}")
    return config

#create neo4j driver (manages db connection) and use to connect to db
def getDriver(config):
    driver = GraphDatabase.driver(
        config['URI'],
        auth=(config['USERNAME'], config['PASSWORD'])
    )
    return driver

#run connection
config = readConfig()
driver = getDriver(config)
print("Connected to Neo4j")

#clear any existing data
def clearDB(session):
    session.run("MATCH (n) DETACH DELETE n")
    print("DB Cleared")


# 1) Knowledge Graph Construction #-------------------------------------------------------------


#Load data from CSV
hotelsDF = pd.read_csv('hotels.csv')
usersDF = pd.read_csv('users.csv')
reviewsDF = pd.read_csv('reviews.csv')
visaDF = pd.read_csv('visa.csv')

#to test if it properly loaded in
print(f"Loaded {len(hotelsDF)} hotels")
print(f"Loaded {len(usersDF)} users")
print(f"Loaded {len(reviewsDF)} reviews")
print(f"Loaded {len(visaDF)} visas")


#Creating nodes -----------
def getCountries(session, hotelsDF, usersDF, visaDF):

    countries = set()  #list of all unique countries in all files (not reviews since it's too expensive and all will br from users or hotels)
    countries.update(usersDF['country'].unique())
    countries.update(hotelsDF['country'].unique())
    countries.update(visaDF['from'].unique())
    countries.update(visaDF['to'].unique())

    for country in countries:
        session.run("""
            UNWIND $names AS name
            MERGE (c:Country {name: name})
            """, names = country )
    print(f"Created {len(countries )} Country nodes")

def getCities(session, hotelsDF):

    cities = set()
    cities.update(hotelsDF['city'].unique())
    for city  in cities:
        session.run("""
            UNWIND $names AS name
            MERGE (c:City {name: name})
            """, names = city )
    print(f"Created {len(cities )} City nodes")

def getHotels(session, hotelsDF):

    #convert DF to list of hotels w/ key-value indexing
    hotels_list = []
    for _, hotel in hotelsDF.iterrows():
        hotels_list.append({
            'hotel_id': int(hotel['hotel_id']),
            'name': hotel['hotel_name'],
            'star_rating': float(hotel['star_rating']) if pd.notna(hotel['star_rating']) else 0.0,
            'cleanliness_base': float(hotel['cleanliness_base']) if pd.notna(hotel['cleanliness_base']) else 0.0,
            'comfort_base': float(hotel['comfort_base']) if pd.notna(hotel['comfort_base']) else 0.0,
            'facilities_base': float(hotel['facilities_base']) if pd.notna(hotel['facilities_base']) else 0.0
        })

    session.run("""
        UNWIND $hotels as hotel
        MERGE (h:Hotel{
            hotel_id: hotel.hotel_id,
            name: hotel.name,
            star_rating: hotel.star_rating,
            cleanliness_base: hotel.cleanliness_base,
            comfort_base: hotel.comfort_base,
            facilities_base: hotel.facilities_base
        })
    """, hotels=hotels_list )
    print(f"Created {len(hotelsDF)} hotel nodes")

def getTraveller(session, usersDf):

    #convert DF to list of users w/ key-value indexing
    users_list = []
    for _, user in usersDF.iterrows():
        users_list.append({
            'user_id': int(user['user_id']),
            'age': user['age_group'] if pd.notna(user['age_group']) else '0',
            'gender': user['user_gender'] if pd.notna(user['user_gender']) else 'Unknown',
            'type': user['traveller_type'] if pd.notna(user['traveller_type']) else 'Unknown',
        })

    session.run("""
        UNWIND $users as user
        MERGE (t:Traveller {
            user_id: user.user_id,
            age: user.age,
            gender: user.gender,
            type: user.type
        })
    """, users=users_list )
    print(f"Created {len(usersDF)} traveller nodes")


def getReviews(session, reviewsDF, batchSize=1000):

    #convert DF to list of reviewas w/ key-value indexing
    totalReviews = len(reviewsDF)

    for i in range(0, totalReviews, batchSize):
        batch = reviewsDF[i:i + batchSize]

        reviews_list = []

        for _, review in batch.iterrows():
            reviews_list.append({
                'review_id': int(review['review_id']),
                'text': review['review_text'] if pd.notna(review['review_text']) else '',
                'date': review['review_date'] if pd.notna(review['review_date']) else '',
                'score_overall': float(review['score_overall']) if pd.notna(review['score_overall']) else 0.0,
                'score_cleanliness': float(review['score_cleanliness']) if pd.notna(review['score_cleanliness']) else 0.0,
                'score_comfort': float(review['score_comfort']) if pd.notna(review['score_comfort']) else 0.0,
                'score_facilities': float(review['score_facilities']) if pd.notna(review['score_facilities']) else 0.0,
                'score_location': float(review['score_location']) if pd.notna(review['score_location']) else 0.0,
                'score_staff': float(review['score_staff']) if pd.notna(review['score_staff']) else 0.0,
                'score_value_for_money': float(review['score_value_for_money']) if pd.notna(review['score_value_for_money']) else 0.0
            })

        session.run("""
            UNWIND $reviews as review
            MERGE (r:Review {
                    review_id: review.review_id,
                    text: review.text,
                    date: review.date,
                    score_overall: review.score_overall,
                    score_cleanliness: review.score_cleanliness,
                    score_comfort: review.score_comfort,
                    score_facilities: review.score_facilities,
                    score_location: review.score_location,
                    score_staff: review.score_staff,
                    score_value_for_money: review.score_value_for_money
            })
        """, reviews=reviews_list )
        print(f"Processed batch {i/1000}")
    print(f"Created {totalReviews} review nodes")

#Creating relationships -----------

def createWROTE(session, reviewsDF, batchSize=1000):

    print("Creating WROTE...")
    #convert DF to list of reviewas w/ key-value indexing
    totalReviews = len(reviewsDF)

    for i in range(0, totalReviews, batchSize):
        batch = reviewsDF[i:i + batchSize]

        relationships = []

        for _, review in batch.iterrows():
            relationships.append({
                'review_id': int(review['review_id']),
                'user_id': int(review['user_id']),
                'hotel_id': int(review['hotel_id'])
            })

        session.run("""
            UNWIND $relations as relation
            MATCH (t: Traveller {user_id: relation.user_id})
            MATCH (r:Review {review_id: relation.review_id})
            MERGE (t)-[:WROTE]->(r)
        """, relations=relationships)
        print(f"Processed batch {i/1000}")

def createFROMCOUNTRY(session, usersDF):

    print("Creating FROMCOUNTRY...")
    for _, user in usersDF.iterrows():
        if pd.notna(user['country']):
            session.run("""
                MATCH (t:Traveller {user_id: $user_id})
                MATCH (c:Country {name: $country_name})
                MERGE (t)-[:FROM_COUNTRY]->(c)
            """, user_id=int(user['user_id']), country_name = user['country'])

def createSTAYEDAT(session, reviewsDF):

    print ("Creating STAYEDAT...")
    reviews = reviewsDF[['user_id', 'hotel_id']].drop_duplicates()

    for _, review in reviews.iterrows():
        session.run("""
            MATCH (t:Traveller {user_id: $user_id})
            MATCH (h:Hotel {hotel_id: $hotel_id})
            MERGE (t)-[:STAYED_AT]->(h)
        """, user_id=int(review['user_id']), hotel_id = review['hotel_id'])

def createREVIEWED(session, reviewsDF, batchSize=1000):

    print("Creating REVIEWED...")
    #convert DF to list of reviewas w/ key-value indexing
    totalReviews = len(reviewsDF)

    for i in range(0, totalReviews, batchSize):
        batch = reviewsDF[i:i + batchSize]

        relationships = []

        for _, review in batch.iterrows():
            relationships.append({
                'review_id': int(review['review_id']),
                'user_id': int(review['user_id']),
                'hotel_id': int(review['hotel_id'])
            })

        session.run("""
            UNWIND $relations as relation
            MATCH (r:Review {review_id: relation.review_id})
            MATCH (h:Hotel {hotel_id: relation.hotel_id})
            MERGE (r)-[:REVIEWED]->(h)
        """, relations=relationships)

        print(f"Processed batch {i/1000}")

def createHotelLOCATEDIN(session, hotelsDF):

    print("Creating Hotel LOCATEDIN...")

    for _, hotel in hotelsDF.iterrows():
        if pd.notna(hotel['city']):
            session.run("""
                MATCH (h:Hotel {hotel_id: $hotel_id})
                MATCH (c:City {name: $city_name})
                MERGE (h)-[:LOCATED_IN]->(c)
            """, hotel_id=int(hotel['hotel_id']), city_name = hotel['city'])

def createCityLOCATEDIN(session, hotelsDF):

    print("Creating City LOCATEDIN...")

    for _, hotel in hotelsDF.iterrows():
        if pd.notna(hotel['city']) and pd.notna(hotel['country']):
            session.run("""
                MATCH (ci:City {name: $city_name})
                MATCH (co:Country {name: $country_name})
                MERGE (ci)-[:LOCATED_IN]->(co)
            """, city_name = hotel['city'], country_name = hotel['country'])


def createNEEDSVISA(session, visaDF):

    print("Creating NEEDSVISA...")

    for _, visa in visaDF.iterrows():
        if pd.notna(visa['from']) and pd.notna(visa['to']):
            if visa['requires_visa'] == 'Yes':
                visa_type = visa['visa_type'] if pd.notna(visa['visa_type']) else 'Unknown'
                session.run("""
                    MATCH (from:Country {name: $from_country})
                    MATCH (to:Country {name: $to_country})
                    MERGE (from)-[:NEEDS_VISA {visa_type: $visa_type}]->(to)
                """, from_country=visa['from'], to_country=visa['to'], visa_type=visa_type)


#run all clear, node, relationship functions using driver
with driver.session() as session:
        clearDB(session)

        #create nodes
        getCountries(session, usersDF, hotelsDF, visaDF)
        getCities(session, hotelsDF)
        getHotels(session, hotelsDF)
        getTraveller(session, usersDF)
        getReviews(session, reviewsDF)

        #create relationships
        createWROTE(session, reviewsDF)
        createFROMCOUNTRY(session, usersDF)
        createSTAYEDAT(session, reviewsDF)
        createREVIEWED(session, reviewsDF)
        createHotelLOCATEDIN(session, hotelsDF)
        createCityLOCATEDIN(session, hotelsDF)
        createNEEDSVISA(session, visaDF)

driver.close()
