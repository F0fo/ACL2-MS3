from neo4j import GraphDatabase
import numpy as np


class GraphRetriever:
    def __init__(self, driver):
        """Initialize with Neo4j driver"""
        self.driver = driver

    def close(self):
        self.driver.close()

    def _extract_graph_data(self, result):
        """Extract nodes and relationships from Neo4j result"""
        nodes = []
        relationships = []

        for record in result:
            for value in record.values():
                if hasattr(value, 'nodes') and hasattr(value, 'relationships'):
                    # It's a path
                    for node in value.nodes:
                        nodes.append({
                            'id': node.element_id,
                            'labels': list(node.labels),
                            'properties': dict(node)
                        })
                    for rel in value.relationships:
                        relationships.append({
                            'id': rel.element_id,
                            'type': rel.type,
                            'start': rel.start_node.element_id,
                            'end': rel.end_node.element_id,
                            'properties': dict(rel)
                        })
                elif hasattr(value, 'labels'):
                    # It's a node
                    nodes.append({
                        'id': value.element_id,
                        'labels': list(value.labels),
                        'properties': dict(value)
                    })

        return {'nodes': nodes, 'relationships': relationships}

    def get_hotels_in_city(self, city):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN path
        """
        with self.driver.session() as session:
            result = session.run(query, city=city)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_in_country(self, country):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name: $country})
        RETURN path
        """
        with self.driver.session() as session:
            result = session.run(query, country=country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_cities_in_country(self, country):
        query = """
        MATCH path = (c:City)-[:LOCATED_IN]->(co:Country {name: $country})
        RETURN path
        """
        with self.driver.session() as session:
            result = session.run(query, country=country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_cities_With_Hotels(self):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)
        RETURN path
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_countries_With_Hotels(self):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        RETURN path
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_by_rating(self, min_rating=5):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.star_rating >= $min_rating
        RETURN path
        ORDER BY h.star_rating DESC
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, min_rating=min_rating)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_in_city_by_rating(self, city, min_rating=5):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        WHERE h.star_rating >= $min_rating
        RETURN path
        ORDER BY h.star_rating DESC
        """
        with self.driver.session() as session:
            result = session.run(query, city=city, min_rating=min_rating)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_in_country_by_rating(self, country, min_rating=5):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name: $country})
        WHERE h.star_rating >= $min_rating
        RETURN path
        ORDER BY h.star_rating DESC
        """
        with self.driver.session() as session:
            result = session.run(query, country=country, min_rating=min_rating)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotel_info(self, hotel_name):
        query = """
        MATCH path = (h:Hotel {name: $hotel_name})-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        RETURN path
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_top_rated_hotels_in_city(self, city, top_n=5):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN path
        ORDER BY h.star_rating DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            result = session.run(query, city=city, top_n=top_n)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_top_rated_hotels_in_country(self, country, top_n=5):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country {name: $country})
        RETURN path
        ORDER BY h.star_rating DESC
        LIMIT $top_n
        """
        with self.driver.session() as session:
            result = session.run(query, country=country, top_n=top_n)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotel_reviews(self, hotel_name):
        randomOffset = np.random.randint(0, 50)
        query = """
        MATCH path = (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel {name: $hotel_name})
        RETURN path
        ORDER BY r.score_overall DESC
        SKIP $randomOffset
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name, randomOffset=randomOffset)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotel_review_count(self, hotel_name):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)
        RETURN COUNT(r) AS review_count
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name)
            return {'query': query.strip(), 'rows': result.data()}

    def get_latest_hotel_reviews(self, hotel_name, limit=5):
        query = """
        MATCH path = (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel {name: $hotel_name})
        RETURN path
        ORDER BY r.date DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name, limit=limit)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotel_reviews_by_travelers_from_country(self, hotel_name, country):
        query = """
        MATCH path = (t:Traveller)-[:FROM_COUNTRY]->(co:Country {name: $country}),
                     (t)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel {name: $hotel_name})
        RETURN path
        ORDER BY r.score_overall DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name, country=country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_with_most_reviews(self, top_n=5):
        query = """
        MATCH path = (h:Hotel)<-[:REVIEWED]-(r:Review)
        WITH h, COUNT(r) AS review_count
        ORDER BY review_count DESC
        LIMIT $top_n
        MATCH p = (h)-[:LOCATED_IN]->(c:City)
        RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query, top_n=top_n)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_countries_requiring_visa(self, from_country):
        query = """
        MATCH path = (tc:Country {name: $from_country})-[:NEEDS_VISA]->(co:Country)
        RETURN path
        """
        with self.driver.session() as session:
            result = session.run(query, from_country=from_country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_countries_not_requiring_visa(self, from_country):
        query = """
        MATCH (tc:Country {name: $from_country})
        MATCH (co:Country)
        WHERE NOT (tc)-[:NEEDS_VISA]->(co) AND tc <> co
        RETURN co
        LIMIT 20
        """
        with self.driver.session() as session:
            result = session.run(query, from_country=from_country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_reviewed_by_travellers_from_country(self, country):
        query = """
        MATCH path = (t:Traveller)-[:FROM_COUNTRY]->(co:Country {name: $country}),
                     (t)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        RETURN path
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, country=country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_reviewed_by_travellers_from_city(self, city):
        query = """
        MATCH path = (ci:City {name: $city})-[:LOCATED_IN]->(c:Country)<-[:FROM_COUNTRY]-(t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        RETURN path
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, city=city)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_accessible_without_visa(self, from_country):
        query = """
        MATCH (tc:Country {name: $from_country})
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE NOT (tc)-[:NEEDS_VISA]->(co)
        RETURN path
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, from_country=from_country)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_average_hotel_rating_by_travellers_from_country(self, hotel_name, country):
        query = """
        MATCH (h:Hotel {name: $hotel_name})<-[:REVIEWED]-(r:Review)<-[:WROTE]-(t:Traveller)-[:FROM_COUNTRY]->(co:Country {name: $country})
        RETURN AVG(r.score_overall) AS average_rating
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name, country=country)
            return {'query': query.strip(), 'rows': result.data()}

    def get_hotel_reviews_filtered(self, hotel_name, gender=None, age_group=None, traveller_type=None):
        query = """
        MATCH path = (t:Traveller)-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel {name: $hotel_name})
        WHERE ($gender IS NULL OR t.gender = $gender)
        AND ($age_group IS NULL OR t.age = $age_group)
        AND ($traveller_type IS NULL OR t.type = $traveller_type)
        RETURN path
        ORDER BY r.score_overall DESC
        LIMIT 10
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_name=hotel_name, gender=gender, age_group=age_group,
                                 traveller_type=traveller_type)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_best_hotels_for_traveller_type(self, traveller_type):
        query = """
        MATCH path = (t:Traveller {type: $traveller_type})-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        WITH h, AVG(r.score_overall) AS score
        ORDER BY score DESC
        LIMIT 10
        MATCH p = (h)-[:LOCATED_IN]->(c:City)
        RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query, traveller_type=traveller_type)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_best_hotels_for_gender(self, gender):
        query = """
        MATCH path = (t:Traveller {gender: $gender})-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        WITH h, AVG(r.score_overall) AS avg_rating
        ORDER BY avg_rating DESC
        LIMIT 10
        MATCH p = (h)-[:LOCATED_IN]->(c:City)
        RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query, gender=gender)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_best_hotels_for_age_group(self, age_group):
        query = """
        MATCH path = (t:Traveller {age: $age_group})-[:WROTE]->(r:Review)-[:REVIEWED]->(h:Hotel)
        WITH h, AVG(r.score_overall) AS avg_rating
        ORDER BY avg_rating DESC
        LIMIT 10
        MATCH p = (h)-[:LOCATED_IN]->(c:City)
        RETURN p
        """
        with self.driver.session() as session:
            result = session.run(query, age_group=age_group)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_with_no_visa_requirements(self):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE NOT EXISTS {
            MATCH (tc:Country)-[:NEEDS_VISA]->(co)
        }
        RETURN path
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_by_cleanliness_base(self, min_cleanliness_base):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE h.cleanliness_base >= $min_cleanliness_base
        RETURN path
        ORDER BY h.cleanliness_base DESC
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, min_cleanliness_base=min_cleanliness_base)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_by_comfort_base(self, min_comfort_base):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE h.comfort_base >= $min_comfort_base
        RETURN path
        ORDER BY h.comfort_base DESC
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, min_comfort_base=min_comfort_base)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_hotels_by_facilities_base(self, min_facilities_base):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE h.facilities_base >= $min_facilities_base
        RETURN path
        ORDER BY h.facilities_base DESC
        LIMIT 25
        """
        with self.driver.session() as session:
            result = session.run(query, min_facilities_base=min_facilities_base)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def compare_two_hotels(self, hotel_name, hotel_name_2):
        query = """
        MATCH path1 = (h1:Hotel {name: $hotel1})-[:LOCATED_IN]->(c1:City),
              path2 = (h2:Hotel {name: $hotel2})-[:LOCATED_IN]->(c2:City)
        RETURN path1, path2
        """
        with self.driver.session() as session:
            result = session.run(query, hotel1=hotel_name, hotel2=hotel_name_2)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}

    def get_all_hotels(self):
        query = """
        MATCH path = (h:Hotel)-[:LOCATED_IN]->(c:City)
        RETURN path
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query)
            return {'query': query.strip(), 'graph': self._extract_graph_data(result)}
