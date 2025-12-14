from neo4j import GraphDatabase
import numpy as np


class GraphRetriever:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # Baseline queries - different forms of queries to retrieve hotel information

    #1. Get hotels based on city
    def get_hotels_in_city(self, city):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN h.name AS hotel, h.star_rating AS rating
        """
        with self.driver.session() as session:
            return session.run(query, city=city).data()
        
    #2. Get hotels based on country    
    def get_hotels_in_country(self, country):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(co:Country {name: $country})
        RETURN h.name AS hotel, h.star_rating AS rating
        """
        with self.driver.session() as session:
            return session.run(query, country=country).data()

    #3. Get cities in a country    
    def get_cities_in_country(self, country):
        query = """
        MATCH (c:City)-[:IN_COUNTRY]->(co:Country {name: $country})
        RETURN c.name AS city
        """
        with self.driver.session() as session:
            return session.run(query, country=country).data()

    #4. Get cities with hotels
    def get_cities_With_Hotels(self):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        RETURN DISTINCT c.name AS city
        """
        with self.driver.session() as session:
            return session.run(query).data()

    #5. Get countries with hotels    
    def get_countries_With_Hotels(self):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(co:Country)
        RETURN DISTINCT co.name AS country
        """
        with self.driver.session() as session:
            return session.run(query).data()
           
    #6. Get hotels by minimum star rating
    def get_hotels_by_rating(self, min_rating):
        query = """
        MATCH (h:Hotel)
        WHERE h.star_rating >= $min_rating
        RETURN h.name AS hotel, h.star_rating AS rating
        ORDER BY h.star_rating DESC
        """
        with self.driver.session() as session:
            return session.run(query, min_rating=min_rating).data()

    #7. Get hotels in a city with minimum star rating    
    def get_hotels_in_city_by_rating(self, city, min_rating):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        WHERE h.star_rating >= $min_rating
        RETURN h.name AS hotel, h.star_rating AS rating
        ORDER BY h.star_rating DESC
        """
        with self.driver.session() as session:
            return session.run(query, city=city, min_rating=min_rating).data()

    #8. Get hotels in a country with minimum star rating
    def get_hotels_in_country_by_rating(self, country, min_rating):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(co:Country {name: $country})
        WHERE h.star_rating >= $min_rating
        RETURN h.name AS hotel, h.star_rating AS rating
        ORDER BY h.star_rating DESC
        """
        with self.driver.session() as session:
            return session.run(query, country=country, min_rating=min_rating).data()    

    #9. Get specific hotel rating    
    def get_hotel_rating(self, hotel_name):
        query = """
        MATCH (h:Hotel {name: $hotel_name})
        RETURN h.name AS hotel, h.star_rating AS rating
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name).data()
        
    #since one hotel per city/country, top rated is same as all hotels in that city/country sorted by rating
    #10. Get top rated hotels in a city
    def get_top_rated_hotels_in_city(self, city, top_n=5):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN h.name AS hotel, h.star_rating AS rating
        ORDER BY h.star_rating DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            return session.run(query, city=city, top_n=top_n).data()

    #11. Get top rated hotels in a country    
    def get_top_rated_hotels_in_country(self, country, top_n=5):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(co:Country {name: $country})
        RETURN h.name AS hotel, h.star_rating AS rating
        ORDER BY h.star_rating DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            return session.run(query, country=country, top_n=top_n).data()

    #12. Get 10 random hotel reviews
    def get_hotel_reviews(self, hotel_name):
        randomOffset = np.random.randint(0,50)
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)
        RETURN r.content AS review, r.rating AS rating
        SKIP $randomOffset
        ORDER BY r.rating DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name).data()    

    #13. Get hotel review count
    def get_hotel_review_count(self, hotel_name):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)
        RETURN COUNT(r) AS review_count
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name).data()

    #14. Get latest hotel reviews    
    def get_latest_hotel_reviews(self, hotel_name, limit=5):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)
        RETURN r.content AS review, r.rating AS rating, r.date AS date
        ORDER BY r.date DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name, limit=limit).data()

    #15. Get specific hotel reviews by travelers from a specific country    
    def get_hotel_reviews_by_travelers_from_country(self, hotel_name, country):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)-[:FROM_COUNTRY]->(co:Country {name: $country})
        RETURN r.content AS review, r.rating AS rating
        ORDER BY r.rating DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name, country=country).data()

    #16. Get hotels with most reviews    
    def get_hotels_with_most_reviews(self, top_n=5):
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)
        RETURN h.name AS hotel, COUNT(r) AS review_count
        ORDER BY review_count DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            return session.run(query, top_n=top_n).data()
        
    #17. Get countries where a traveller from a specific country needs a visa
    def get_countries_requiring_visa(self, from_country):
        query = """
        MATCH (t:Traveller)-[:FROM_COUNTRY]->(tc:Country {name: $from_country})-[:NEEDS_VISA]->(co:Country)
        RETURN co.name AS country
        """
        with self.driver.session() as session:
            return session.run(query, traveller_country=from_country).data()
        
    #18. Get countries where a traveller from a specific country does not need a visa
    def get_countries_not_requiring_visa(self, from_country):
        query = """
        MATCH (t:Traveller)-[:FROM_COUNTRY]->(tc:Country {name: $from_country})
        MATCH (co:Country)
        WHERE NOT (tc)-[:NEEDS_VISA]->(co)
        RETURN co.name AS country
        """
        with self.driver.session() as session:
            return session.run(query, traveller_country=from_country).data()


    # ------------------ case edging queries ------------------    
    #19. get hotels reviewed by travellers from a specific country
    def get_hotels_reviewed_by_travellers_from_country(self, country):
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)-[:FROM_COUNTRY]->(co:Country {name: $country})
        RETURN DISTINCT h.name AS hotel
        """
        with self.driver.session() as session:
            return session.run(query, country=country).data()
        
    #20. get hotels reviewed by travellers from a specific city
    def get_hotels_reviewed_by_travellers_from_city(self, city):
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)-[:FROM_CITY]->(ci:City {name: $city})
        RETURN DISTINCT h.name AS hotel
        """
        with self.driver.session() as session:
            return session.run(query, city=city).data()
    
    #22. get hotels where traveller from a specific country can stay without visa
    def get_hotels_accessible_without_visa(self, from_country):
        query = """
        MATCH (t:Traveller)-[:FROM_COUNTRY]->(tc:Country {name: $from_country})
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(co:Country)
        WHERE NOT (tc)-[:NEEDS_VISA]->(co)
        RETURN DISTINCT h.name AS hotel
        """
        with self.driver.session() as session:
            return session.run(query, traveller_country=from_country).data()
        
    #23. get average hotel rating by travellers from a specific country
    def get_average_hotel_rating_by_travellers_from_country(self, hotel_name, country):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)-[:FROM_COUNTRY]->(co:Country {name: $country})
        RETURN AVG(r.rating) AS average_rating
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name, country=country).data()
        
    #21. Filter hotels reviews by demographic
    def get_hotel_reviews_filtered(self, hotel_name, gender=None, age_group=None, traveller_type=None):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
        WHERE ($gender IS NULL OR t.gender = $gender)
        AND ($age_group IS NULL OR t.age_group = $age_group)
        AND ($traveller_type IS NULL OR t.traveller_type = $traveller_type)
        RETURN r.content AS review, r.rating AS rating
        ORDER BY r.rating DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            return session.run(query, hotel_name=hotel_name, gender=gender, age_group=age_group, traveller_type=traveller_type).data()  

    #25. get best hotels for a specific traveller type
    def get_best_hotels_for_traveller_type(self, traveller_type):
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)
        WHERE t.traveller_type = $traveller_type
        RETURN h.name AS hotel, AVG(r.rating) AS score
        ORDER BY score DESC
        LIMIT 10
        """

        with self.driver.session() as session:
            return session.run(query, traveller_type=traveller_type).data()
    
    #26. get best hotels for specific gender of travellers
    def get_best_hotels_for_gender(self, gender):
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller {gender: $gender}) 
        RETURN h.name AS hotel, AVG(r.rating) AS avg_rating
        ORDER BY avg_rating DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            return session.run(query, gender=gender).data()
        
    #27. get best hotels for specific age group of travellers
    def get_best_hotels_for_age_group(self, age_group):
        query = """
        MATCH (h:Hotel)<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller {age_group: $age_group}) 
        RETURN h.name AS hotel, AVG(r.rating) AS avg_rating
        ORDER BY avg_rating DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            return session.run(query, age_group=age_group).data()  

    #28. hotels with no visa requirements
    def get_hotels_with_no_visa_requirements(self):
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:IN_COUNTRY]->(co:Country)
        WHERE NOT EXISTS {
            MATCH (t:Traveller)-[:FROM_COUNTRY]->(tc:Country)-[:NEEDS_VISA]->(co)
        }
        RETURN DISTINCT h.name AS hotel
        """
        with self.driver.session() as session:
            return session.run(query).data()   
        
    #29. get hotels based on one of it's cleanliness base
    def get_hotels_by_cleanliness_base(self, min_cleanliness_base):
        query = """
        MATCH (h:Hotel)
        WHERE h.cleanliness_rating >= $min_cleanliness_base
        RETURN h.name AS hotel, h.cleanliness_base AS cleanliness_base
        ORDER BY h.cleanliness_rating DESC
        """
        with self.driver.session() as session:
            return session.run(query, min_cleanliness_base=min_cleanliness_base).data()
    
    def get_hotels_by_comfort_base(self, min_comfort_base):
        query = """
        MATCH (h:Hotel)
        WHERE h.comfort_rating >= $min_comfort_base
        RETURN h.name AS hotel, h.comfort_base AS comfort_base
        ORDER BY h.comfort_rating DESC
        """
        with self.driver.session() as session:
            return session.run(query, min_comfort_base=min_comfort_base).data()
        
    def get_hotels_by_facilities_base(self, min_facilities_base):
        query = """
        MATCH (h:Hotel)
        WHERE h.facilities_rating >= $min_facilities_base
        RETURN h.name AS hotel, h.facilities_base AS facilities_base
        ORDER BY h.facilities_rating DESC
        """
        with self.driver.session() as session:
            return session.run(query, min_facilities_base=min_facilities_base).data()

    #30. Compare two hotels
    def compare_two_hotels(self, hotel_name, hotel_name_2):
        query = """
        MATCH (h1:Hotel {name: $hotel1}), (h2:Hotel {name: $hotel2})
        RETURN h1,
               h2
        """
        with self.driver.session() as session:
            return session.run(query, hotel1=hotel_name, hotel2=hotel_name_2).data()
        