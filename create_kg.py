"""
Knowledge Graph Construction Script
Creates nodes and relationships in Neo4j from CSV data
"""

from neo4j import GraphDatabase
import pandas as pd
from db_manager import DBManager


class KnowledgeGraphBuilder:
    """Builds knowledge graph from travel review data"""
    
    def __init__(self, driver):
        """Initialize with Neo4j driver"""
        self.driver = driver
        
        # Load data from CSV files
        print("\n Loading data from CSV files...")
        self.hotelsDF = pd.read_csv('./data/hotels.csv')
        self.usersDF = pd.read_csv('./data/users.csv')
        self.reviewsDF = pd.read_csv('./data/reviews.csv')
        self.visaDF = pd.read_csv('./data/visa.csv')
        
        print(f"  ✓ Loaded {len(self.hotelsDF):,} hotels")
        print(f"  ✓ Loaded {len(self.usersDF):,} users")
        print(f"  ✓ Loaded {len(self.reviewsDF):,} reviews")
        print(f"  ✓ Loaded {len(self.visaDF):,} visa records")
    
    # ==================== NODE CREATION METHODS ====================
    
    def create_countries(self):
        """Create Country nodes from all data sources"""
        
        countries = set()
        countries.update(self.usersDF['country'].dropna().unique())
        countries.update(self.hotelsDF['country'].dropna().unique())
        countries.update(self.visaDF['from'].dropna().unique())
        countries.update(self.visaDF['to'].dropna().unique())
        
        countries_list = list(countries)
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $names AS name
                MERGE (c:Country {name: name})
            """, names=countries_list)
        
        print(f"  ✓ Created {len(countries_list)} Country nodes")
    
    def create_cities(self):
        """Create City nodes from hotels data"""
        
        cities = self.hotelsDF['city'].dropna().unique()
        cities_list = list(cities)
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $names AS name
                MERGE (c:City {name: name})
            """, names=cities_list)
        
        print(f"  ✓ Created {len(cities_list)} City nodes")
    
    def create_hotels(self):
        """Create Hotel nodes from hotels data"""
        
        hotels_list = []
        for _, hotel in self.hotelsDF.iterrows():
            hotels_list.append({
                'hotel_id': int(hotel['hotel_id']),
                'name': hotel['hotel_name'],
                'star_rating': float(hotel['star_rating']) if pd.notna(hotel['star_rating']) else 0.0,
                'cleanliness_base': float(hotel['cleanliness_base']) if pd.notna(hotel['cleanliness_base']) else 0.0,
                'comfort_base': float(hotel['comfort_base']) if pd.notna(hotel['comfort_base']) else 0.0,
                'facilities_base': float(hotel['facilities_base']) if pd.notna(hotel['facilities_base']) else 0.0
            })
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $hotels as hotel
                MERGE (h:Hotel {hotel_id: hotel.hotel_id})
                SET h.name = hotel.name,
                    h.star_rating = hotel.star_rating,
                    h.cleanliness_base = hotel.cleanliness_base,
                    h.comfort_base = hotel.comfort_base,
                    h.facilities_base = hotel.facilities_base
            """, hotels=hotels_list)
        
        print(f"  ✓ Created {len(hotels_list)} Hotel nodes")
    
    def create_travellers(self):
        """Create Traveller nodes from users data"""
        
        users_list = []
        for _, user in self.usersDF.iterrows():
            users_list.append({
                'user_id': int(user['user_id']),
                'age': user['age_group'] if pd.notna(user['age_group']) else 'Unknown',
                'gender': user['user_gender'] if pd.notna(user['user_gender']) else 'Unknown',
                'type': user['traveller_type'] if pd.notna(user['traveller_type']) else 'Unknown',
            })
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $users as user
                MERGE (t:Traveller {user_id: user.user_id})
                SET t.age = user.age,
                    t.gender = user.gender,
                    t.type = user.type
            """, users=users_list)
        
        print(f"  ✓ Created {len(users_list)} Traveller nodes")
    
    def create_reviews(self, batch_size=1000):
        """Create Review nodes from reviews data"""
        
        total_reviews = len(self.reviewsDF)
        
        for i in range(0, total_reviews, batch_size):
            batch = self.reviewsDF[i:i + batch_size]
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
            
            with self.driver.session() as session:
                session.run("""
                    UNWIND $reviews as review
                    MERGE (r:Review {review_id: review.review_id})
                    SET r.text = review.text,
                        r.date = review.date,
                        r.score_overall = review.score_overall,
                        r.score_cleanliness = review.score_cleanliness,
                        r.score_comfort = review.score_comfort,
                        r.score_facilities = review.score_facilities,
                        r.score_location = review.score_location,
                        r.score_staff = review.score_staff,
                        r.score_value_for_money = review.score_value_for_money
                """, reviews=reviews_list)
            
            print(f" Processed batch {i//batch_size + 1}/{(total_reviews + batch_size - 1)//batch_size}")
        
        print(f"  ✓ Created {total_reviews:,} Review nodes")
    
    # ==================== RELATIONSHIP CREATION METHODS ====================
    
    def create_wrote_relationships(self, batch_size=1000):
        """Create WROTE relationships between Traveller and Review"""
        
        total_reviews = len(self.reviewsDF)
        
        for i in range(0, total_reviews, batch_size):
            batch = self.reviewsDF[i:i + batch_size]
            relationships = []
            
            for _, review in batch.iterrows():
                relationships.append({
                    'review_id': int(review['review_id']),
                    'user_id': int(review['user_id'])
                })
            
            with self.driver.session() as session:
                session.run("""
                    UNWIND $relations as relation
                    MATCH (t:Traveller {user_id: relation.user_id})
                    MATCH (r:Review {review_id: relation.review_id})
                    MERGE (t)-[:WROTE]->(r)
                """, relations=relationships)
            
            print(f"  Processed batch {i//batch_size + 1}/{(total_reviews + batch_size - 1)//batch_size}")
        
        print(f"  ✓ Created WROTE relationships")
    
    def create_from_country_relationships(self):
        """Create FROM_COUNTRY relationships between Traveller and Country"""
        
        relationships = []
        for _, user in self.usersDF.iterrows():
            if pd.notna(user['country']):
                relationships.append({
                    'user_id': int(user['user_id']),
                    'country_name': user['country']
                })
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $relations as relation
                MATCH (t:Traveller {user_id: relation.user_id})
                MATCH (c:Country {name: relation.country_name})
                MERGE (t)-[:FROM_COUNTRY]->(c)
            """, relations=relationships)
        
        print(f"  ✓ Created {len(relationships)} FROM_COUNTRY relationships")
    
    def create_stayed_at_relationships(self, batch_size=1000):
        """Create STAYED_AT relationships between Traveller and Hotel"""
        
        # Get unique user-hotel pairs
        reviews = self.reviewsDF[['user_id', 'hotel_id']].drop_duplicates()
        total = len(reviews)
        
        for i in range(0, total, batch_size):
            batch = reviews[i:i + batch_size]
            relationships = []
            
            for _, review in batch.iterrows():
                relationships.append({
                    'user_id': int(review['user_id']),
                    'hotel_id': int(review['hotel_id'])
                })
            
            with self.driver.session() as session:
                session.run("""
                    UNWIND $relations AS relation
                    MATCH (t:Traveller {user_id: relation.user_id})
                    MATCH (h:Hotel {hotel_id: relation.hotel_id})
                    MERGE (t)-[:STAYED_AT]->(h)
                """, relations=relationships)
            
            print(f"   Processed batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
        
        print(f"  ✓ Created STAYED_AT relationships")
    
    def create_reviewed_relationships(self, batch_size=1000):
        """Create REVIEWED relationships between Review and Hotel"""
        
        total_reviews = len(self.reviewsDF)
        
        for i in range(0, total_reviews, batch_size):
            batch = self.reviewsDF[i:i + batch_size]
            relationships = []
            
            for _, review in batch.iterrows():
                relationships.append({
                    'review_id': int(review['review_id']),
                    'hotel_id': int(review['hotel_id'])
                })
            
            with self.driver.session() as session:
                session.run("""
                    UNWIND $relations as relation
                    MATCH (r:Review {review_id: relation.review_id})
                    MATCH (h:Hotel {hotel_id: relation.hotel_id})
                    MERGE (r)-[:REVIEWED]->(h)
                """, relations=relationships)
            
            print(f"  ⏳ Processed batch {i//batch_size + 1}/{(total_reviews + batch_size - 1)//batch_size}")
        
        print(f"  ✓ Created REVIEWED relationships")
    
    def create_hotel_located_in_relationships(self):
        """Create LOCATED_IN relationships between Hotel and City"""
        
        relationships = []
        for _, hotel in self.hotelsDF.iterrows():
            if pd.notna(hotel['city']):
                relationships.append({
                    'hotel_id': int(hotel['hotel_id']),
                    'city_name': hotel['city']
                })
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $relations AS relation
                MATCH (h:Hotel {hotel_id: relation.hotel_id})
                MATCH (c:City {name: relation.city_name})
                MERGE (h)-[:LOCATED_IN]->(c)
            """, relations=relationships)
        
        print(f"  ✓ Created {len(relationships)} Hotel LOCATED_IN relationships")
    
    def create_city_located_in_relationships(self):
        """Create LOCATED_IN relationships between City and Country"""
        
        # Get unique city-country pairs
        city_country_pairs = self.hotelsDF[['city', 'country']].drop_duplicates()
        relationships = []
        
        for _, row in city_country_pairs.iterrows():
            if pd.notna(row['city']) and pd.notna(row['country']):
                relationships.append({
                    'city_name': row['city'],
                    'country_name': row['country']
                })
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $relations AS relation
                MATCH (ci:City {name: relation.city_name})
                MATCH (co:Country {name: relation.country_name})
                MERGE (ci)-[:LOCATED_IN]->(co)
            """, relations=relationships)
        
        print(f"  ✓ Created {len(relationships)} City LOCATED_IN relationships")
    
    def create_needs_visa_relationships(self):
        """Create NEEDS_VISA relationships between Countries"""
        
        relationships = []
        for _, visa in self.visaDF.iterrows():
            if pd.notna(visa['from']) and pd.notna(visa['to']) and visa['requires_visa'] == 'Yes':
                visa_type = visa['visa_type'] if pd.notna(visa['visa_type']) else 'Unknown'
                relationships.append({
                    'from_country': visa['from'],
                    'to_country': visa['to'],
                    'visa_type': visa_type
                })
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $relations AS relation
                MATCH (from:Country {name: relation.from_country})
                MATCH (to:Country {name: relation.to_country})
                MERGE (from)-[:NEEDS_VISA {visa_type: relation.visa_type}]->(to)
            """, relations=relationships)
        
        print(f"  ✓ Created {len(relationships)} NEEDS_VISA relationships")
    
    # ==================== COMBINED METHODS ====================
    
    def create_all_nodes(self):
        """Create all node types"""
        print("\n" + "="*60)
        print("CREATING ALL NODES")
        print("="*60)
        
        self.create_countries()
        self.create_cities()
        self.create_hotels()
        self.create_travellers()
        self.create_reviews()
        
        print("\n All nodes created successfully!")
    
    def create_all_relationships(self):
        """Create all relationship types"""
        print("\n" + "="*60)
        print("CREATING ALL RELATIONSHIPS")
        print("="*60)
        
        self.create_wrote_relationships()
        self.create_from_country_relationships()
        self.create_stayed_at_relationships()
        self.create_reviewed_relationships()
        self.create_hotel_located_in_relationships()
        self.create_city_located_in_relationships()
        self.create_needs_visa_relationships()
        
        print("\n All relationships created successfully!")
    
    def build_complete_graph(self, clear_existing=False):
        """Build the complete knowledge graph"""
        print("\n" + "="*60)
        print("BUILDING COMPLETE KNOWLEDGE GRAPH")
        print("="*60)
        
        if clear_existing:
            print("\n  Clearing existing database...")
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("  ✓ Database cleared")
        
        self.create_all_nodes()
        self.create_all_relationships()
        
        print("\n" + "="*60)
        print(" KNOWLEDGE GRAPH CONSTRUCTION COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Neo4j Knowledge Graph from Travel Data')
    parser.add_argument('--clear', action='store_true', help='Clear database before building')
    parser.add_argument('--nodes-only', action='store_true', help='Only create nodes')
    parser.add_argument('--relationships-only', action='store_true', help='Only create relationships')
    parser.add_argument('--check', action='store_true', help='Check database status only')
    
    args = parser.parse_args()
    
    try:
        # Initialize database manager
        db_manager = DBManager()
        
        # If just checking status
        if args.check:
            db_manager.print_status()
            db_manager.close()
            exit(0)
        
        # Initialize knowledge graph builder
        kg_builder = KnowledgeGraphBuilder(db_manager.driver)
        
        # Execute based on arguments
        if args.nodes_only:
            if args.clear:
                db_manager.clear_db()
            kg_builder.create_all_nodes()
        elif args.relationships_only:
            kg_builder.create_all_relationships()
        else:
            # Full build
            kg_builder.build_complete_graph(clear_existing=args.clear)
        
        # Show final status
        print("\n")
        db_manager.print_status()
        
        # Close connection
        db_manager.close()
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()