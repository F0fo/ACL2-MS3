
import os
import pickle
import faiss
import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from db_manager import DBManager
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge


# ==============================================================================
# NODE2VEC EMBEDDING RETRIEVER (Original Implementation)
# ==============================================================================

class EmbeddingRetriever:

    # Configuration
    GRAPH_NAME = "travelGraph"
    EMBEDDING_PROP = "node2vecEmbedding"
    EMBEDDING_DIM = 128
    VECTOR_INDEX = "hotel_node2vec_index"

    FAISS_INDEX_PATH = "faiss_hotel_index.bin"
    HOTEL_MAPPING_PATH = "hotel_mapping.pkl"


    def __init__(self, driver):
        self.driver = driver

        # FAISS index and mappings (initialized when built or loaded)
        self.faiss_index = None
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}


    def create_graph(self):

        drop_query = f"""
        CALL gds.graph.drop('{self.GRAPH_NAME}', false)
        YIELD graphName
        """

        create_query = f"""
        CALL gds.graph.project(
          '{self.GRAPH_NAME}',
          ['Hotel', 'City', 'Country', 'Traveller', 'Review'],
          {{
            LOCATED_IN: {{orientation: 'UNDIRECTED'}},
            REVIEWED: {{orientation: 'UNDIRECTED'}},
            WROTE: {{orientation: 'UNDIRECTED'}},
            FROM_COUNTRY: {{orientation: 'UNDIRECTED'}},
            STAYED_AT: {{orientation: 'UNDIRECTED'}}
          }}
        )
        """

        with self.driver.session() as session:
            # Try to drop existing graph
            try:
                session.run(drop_query)
                print("  Dropped existing graph projection")
            except ClientError as e:
                if "not found" not in str(e).lower():
                    print(f"Warning: Could not drop graph: {e}")

            # Create new graph projection
            try:
                result = session.run(create_query)
                data = result.data()[0]
                print(f"  Graph created: {data['nodeCount']} nodes, {data['relationshipCount']} relationships")
                return data
            except ClientError as e:
                print(f" Error creating graph: {e}")
                raise

    def run_node2vec(self):
        print("\n Running Node2Vec algorithm...")

        query = f"""
        CALL gds.node2vec.write(
          '{self.GRAPH_NAME}',
          {{
            writeProperty: '{self.EMBEDDING_PROP}',
            embeddingDimension: 128,
            walkLength: 40,
            iterations: 10,
            walkBufferSize: 1000,
            randomSeed: 42
          }}
        )
        """

        with self.driver.session() as session:
            try:
                result = session.run(query)
                data = result.data()[0]
                print(f"  Node2Vec complete:")
                print(f"    - Nodes processed: {data.get('nodePropertiesWritten', 'N/A')}")
                print(f"    - Computation time: {data.get('computeMillis', 0)/1000:.2f}s")
                return data
            except ClientError as e:
                print(f" Error running Node2Vec: {e}")
                raise

    def create_vector_index(self):
        """Create vector index for similarity search"""
        print("\n Creating vector index...")

        drop_query = f"""
        DROP INDEX {self.VECTOR_INDEX} IF EXISTS
        """

        create_query = f"""
        CREATE VECTOR INDEX {self.VECTOR_INDEX} IF NOT EXISTS
        FOR (h:Hotel)
        ON (h.{self.EMBEDDING_PROP})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: 128,
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """

        with self.driver.session() as session:
            try:
                # Drop existing index
                session.run(drop_query)

                # Create new index
                session.run(create_query)
                print(f"  Vector index '{self.VECTOR_INDEX}' created")
            except ClientError as e:
                print(f"  Error creating vector index: {e}")
                raise


    def verify_embeddings_exist(self):
        """Check if embeddings have been created for hotels"""
        print("\n Verifying embeddings...")

        query = f"""
        MATCH (h:Hotel)
        RETURN
            count(h) as total_hotels,
            count(h.{self.EMBEDDING_PROP}) as hotels_with_embeddings
        """

        with self.driver.session() as session:
            try:
                result = session.run(query)
                data = result.single()
                total = data['total_hotels']
                with_embeddings = data['hotels_with_embeddings']

                print(f" Hotels with embeddings: {with_embeddings}/{total}")

                if with_embeddings == 0:
                    print("No embeddings found!")
                    return False
                elif with_embeddings < total:
                    print(f"Warning: {total - with_embeddings} hotels missing embeddings")
                else:
                    print(" All hotels have embeddings")

                return with_embeddings > 0

            except Exception as e:
                print(f" Error verifying embeddings: {e}")
                return False

    def build_faiss_index(self):
        """Build FAISS index from Node2Vec embeddings stored in Neo4j"""
        print("\n Building FAISS index...")

        query = f"""
        MATCH (h:Hotel)
        WHERE h.{self.EMBEDDING_PROP} IS NOT NULL
        RETURN h.hotel_id as hotel_id, h.{self.EMBEDDING_PROP} as embedding
        ORDER BY hotel_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()

        if not records:
            print("  No hotels with embeddings found!")
            return False

        print(f"  Found {len(records)} hotels with embeddings")

        # Build embedding matrix and mappings
        embeddings_list = []
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}

        for idx, record in enumerate(records):
            hotel_id = record['hotel_id']
            embedding = np.array(record['embedding'], dtype='float32')

            embeddings_list.append(embedding)
            self.hotel_id_to_idx[hotel_id] = idx
            self.idx_to_hotel_id[idx] = hotel_id

        # Stack into matrix
        embeddings_matrix = np.vstack(embeddings_list)
        print(f"  Embeddings matrix shape: {embeddings_matrix.shape}")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)

        # Create FAISS index (Inner Product = cosine similarity after normalization)
        self.faiss_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.faiss_index.add(embeddings_matrix)

        print(f"  FAISS index built with {self.faiss_index.ntotal} vectors")
        return True

    def save_faiss_index(self):
        """Save FAISS index and mappings to disk"""
        print("\n Saving FAISS index...")

        if self.faiss_index is None:
            print("  No FAISS index to save!")
            return False

        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, self.FAISS_INDEX_PATH)
            print(f"  Index saved to {self.FAISS_INDEX_PATH}")

            # Save mappings
            with open(self.HOTEL_MAPPING_PATH, 'wb') as f:
                pickle.dump({
                    'hotel_id_to_idx': self.hotel_id_to_idx,
                    'idx_to_hotel_id': self.idx_to_hotel_id
                }, f)
            print(f"  Mappings saved to {self.HOTEL_MAPPING_PATH}")

            return True
        except Exception as e:
            print(f"  Error saving: {e}")
            return False

    def load_faiss_index(self):
        """Load FAISS index and mappings from disk"""
        print("\n Loading FAISS index...")

        if not os.path.exists(self.FAISS_INDEX_PATH):
            print(f"  Index file not found: {self.FAISS_INDEX_PATH}")
            return False

        if not os.path.exists(self.HOTEL_MAPPING_PATH):
            print(f"  Mapping file not found: {self.HOTEL_MAPPING_PATH}")
            return False

        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(self.FAISS_INDEX_PATH)
            print(f"  Index loaded ({self.faiss_index.ntotal} vectors)")

            # Load mappings
            with open(self.HOTEL_MAPPING_PATH, 'rb') as f:
                mappings = pickle.load(f)
            self.hotel_id_to_idx = mappings['hotel_id_to_idx']
            self.idx_to_hotel_id = mappings['idx_to_hotel_id']
            print(f"  Mappings loaded ({len(self.hotel_id_to_idx)} hotels)")

            return True
        except Exception as e:
            print(f"  Error loading: {e}")
            return False

    def find_similar_hotels(self, hotel_name, top_k=5):
        """Find similar hotels using Node2Vec embeddings"""
        print(f"\n Searching similar hotels to: '{hotel_name}'")

        # Ensure FAISS index is loaded
        if self.faiss_index is None:
            print("  No FAISS index loaded. Attempting to load...")
            if not self.load_faiss_index():
                print("  Failed to load FAISS index. Run setup first.")
                return []

        query = f"""
        MATCH (h:Hotel {{name: $hotel}})
        RETURN h.hotel_id AS hotel_id
        """

        with self.driver.session() as session:
            result = session.run(query, hotel=hotel_name)
            record = result.single()

            if not record:
                print(f"  Hotel '{hotel_name}' not found in database")
                return []
            hotel_id = record['hotel_id']

        # Get position of retrieved hotel in FAISS index
        if hotel_id not in self.hotel_id_to_idx:
            print(f"  Hotel '{hotel_name}' does not have an embedding")
            return []

        idx = self.hotel_id_to_idx[hotel_id]

        #get query vector
        query_vector = self.faiss_index.reconstruct(idx)
        query_vector = query_vector.reshape(1, -1)

        #search for similar vectors in faiss index
        D, I = self.faiss_index.search(query_vector, top_k + 1)  # +1 to skip itself

        similar_hotels = []
        scores = []

        for i, score in zip(I[0], D[0]):
            result_hotel_id = self.idx_to_hotel_id[i]
            if result_hotel_id == hotel_id:
                continue  #skip itself
            similar_hotels.append(result_hotel_id)
            scores.append(score)

            if len(similar_hotels) >= top_k:
                break

        query = """
        UNWIND $hotel_ids as hotel_id
        MATCH (h:Hotel {hotel_id: hotel_id})
        RETURN h.hotel_id as hotel_id, h.name as name
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_ids=similar_hotels)
            hotel_details = {record['hotel_id']: record for record in result.data()}

        final_results = []
        for hid, score in zip(similar_hotels, scores):
            if hid in hotel_details:
                final_results.append({
                    'hotel': hotel_details[hid]['name'],
                    'hotel_id': hid,
                    'score': score
                })
        return final_results

    def get_hotel_by_id(self, hotel_id):
        """Get hotel name by ID"""
        query = """
        MATCH (h:Hotel {hotel_id: $hotel_id})
        RETURN h.name as name
        """

        with self.driver.session() as session:
            result = session.run(query, hotel_id=hotel_id)
            record = result.single()
            return record['name'] if record else None

    def get_random_hotel(self):
        """Get a random hotel from the database"""
        query = f"""
        MATCH (h:Hotel)
        WHERE h.{self.EMBEDDING_PROP} IS NOT NULL
        RETURN h.name as name, h.hotel_id as hotel_id
        ORDER BY rand()
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return record['name'], record['hotel_id']
            return None, None

    def setup_embeddings(self):
        """Complete setup process: create graph, run node2vec, build FAISS index"""
        print("\n" + "="*60)
        print("SETTING UP NODE2VEC EMBEDDINGS")
        print("="*60)

        try:
            # Step 1: Create graph projection
            self.create_graph()

            # Step 2: Run Node2Vec
            self.run_node2vec()

            # Step 3: Verify embeddings
            if not self.verify_embeddings_exist():
                print("\n Setup failed: No embeddings created")
                return False

            # Step 4: Create Neo4j vector index (optional, for native Neo4j search)
            self.create_vector_index()

            # Step 5: Build FAISS index for fast similarity search
            if not self.build_faiss_index():
                print("\n Setup failed: Could not build FAISS index")
                return False

            # Step 6: Save FAISS index to disk
            if not self.save_faiss_index():
                print("\n Warning: Could not save FAISS index")

            print("\n" + "="*60)
            print(" NODE2VEC EMBEDDING SETUP COMPLETE")
            print("="*60)
            return True

        except Exception as e:
            print(f"\nSetup failed: {e}")
            return False


# ==============================================================================
# FEATURE EMBEDDING RETRIEVER (Enriched Hotel Properties)
# ==============================================================================

class FeatureEmbeddingRetriever:
    """
    Feature-based Embedding Retriever

    Creates embeddings from enriched hotel properties:
    - Base properties: star_rating, cleanliness_base, comfort_base, facilities_base
    - Location: city_id, country_id (encoded)
    - Computed from reviews: review_count, avg_reviewer_score, avg_cleanliness,
      avg_comfort, avg_facilities, avg_location, avg_staff, avg_value

    This approach contrasts with Node2Vec:
    - Node2Vec: Graph STRUCTURE (how hotels connect via reviews/travelers)
    - Feature Embedding: Hotel PROPERTIES (ratings, scores, aggregated review data)
    """

    EMBEDDING_PROP = "featureEmbedding"
    FAISS_INDEX_PATH = "faiss_hotel_features.bin"
    HOTEL_MAPPING_PATH = "hotel_mapping_features.pkl"

    # Properties used for feature embedding (after enrichment)
    FEATURE_PROPERTIES = [
        # Base properties
        'star_rating',
        'cleanliness_base',
        'comfort_base',
        'facilities_base',
        # Computed from reviews
        'review_count',
        'avg_reviewer_score',
        'avg_cleanliness',
        'avg_comfort',
        'avg_facilities',
        'avg_location',
        'avg_staff',
        'avg_value'
    ]

    def __init__(self, driver):
        self.driver = driver

        # FAISS index and mappings
        self.faiss_index = None
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}
        self.embedding_dim = len(self.FEATURE_PROPERTIES)

    def enrich_hotel_properties(self):
        """
        Compute and store enriched properties on Hotel nodes:
        - review_count: Number of reviews for this hotel
        - avg_reviewer_score: Average overall score from reviews
        - avg_cleanliness, avg_comfort, avg_facilities, avg_location, avg_staff, avg_value
        """
        print("\n Enriching Hotel nodes with computed properties...")

        # Compute aggregated review statistics for each hotel
        # Note: Review properties are named score_overall, score_cleanliness, etc.
        query = """
        MATCH (h:Hotel)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WITH h,
             count(r) as review_count,
             avg(r.score_overall) as avg_reviewer_score,
             avg(r.score_cleanliness) as avg_cleanliness,
             avg(r.score_comfort) as avg_comfort,
             avg(r.score_facilities) as avg_facilities,
             avg(r.score_location) as avg_location,
             avg(r.score_staff) as avg_staff,
             avg(r.score_value_for_money) as avg_value
        SET h.review_count = coalesce(review_count, 0),
            h.avg_reviewer_score = coalesce(avg_reviewer_score, 0),
            h.avg_cleanliness = coalesce(avg_cleanliness, 0),
            h.avg_comfort = coalesce(avg_comfort, 0),
            h.avg_facilities = coalesce(avg_facilities, 0),
            h.avg_location = coalesce(avg_location, 0),
            h.avg_staff = coalesce(avg_staff, 0),
            h.avg_value = coalesce(avg_value, 0)
        RETURN count(h) as hotels_updated
        """

        with self.driver.session() as session:
            try:
                result = session.run(query)
                data = result.single()
                print(f"  Enriched {data['hotels_updated']} hotels with review statistics")

                # Print sample to verify
                sample = session.run("""
                    MATCH (h:Hotel)
                    RETURN h.name as name, h.review_count as reviews,
                           h.avg_reviewer_score as avg_score
                    LIMIT 3
                """).data()
                print("  Sample enriched data:")
                for s in sample:
                    print(f"    - {s['name']}: {s['reviews']} reviews, avg score: {s['avg_score']:.2f}")

                return True
            except Exception as e:
                print(f"  Error enriching hotels: {e}")
                return False

    def get_enriched_properties(self):
        """Retrieve enriched properties for all hotels"""
        print("\n Retrieving enriched hotel properties...")

        query = """
        MATCH (h:Hotel)
        RETURN h.hotel_id as hotel_id,
               h.name as name,
               h.star_rating as star_rating,
               h.cleanliness_base as cleanliness_base,
               h.comfort_base as comfort_base,
               h.facilities_base as facilities_base,
               h.review_count as review_count,
               h.avg_reviewer_score as avg_reviewer_score,
               h.avg_cleanliness as avg_cleanliness,
               h.avg_comfort as avg_comfort,
               h.avg_facilities as avg_facilities,
               h.avg_location as avg_location,
               h.avg_staff as avg_staff,
               h.avg_value as avg_value
        ORDER BY h.hotel_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()
            print(f"  Retrieved {len(records)} hotels")
            return records

    def build_feature_embeddings(self):
        """Build feature embeddings from enriched properties"""
        print("\n Building feature embeddings...")

        records = self.get_enriched_properties()

        if not records:
            print("  No hotel data found!")
            return False

        embeddings_list = []
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}

        for idx, record in enumerate(records):
            hotel_id = record['hotel_id']

            # Create feature vector from properties
            features = []
            for prop in self.FEATURE_PROPERTIES:
                value = record.get(prop, 0)
                features.append(float(value) if value is not None else 0.0)

            embedding = np.array(features, dtype='float32')
            embeddings_list.append(embedding)

            self.hotel_id_to_idx[hotel_id] = idx
            self.idx_to_hotel_id[idx] = hotel_id

        # Stack into matrix
        embeddings_matrix = np.vstack(embeddings_list)
        print(f"  Raw embeddings matrix shape: {embeddings_matrix.shape}")

        # Normalize each feature column (min-max scaling)
        for i in range(embeddings_matrix.shape[1]):
            col = embeddings_matrix[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                embeddings_matrix[:, i] = (col - min_val) / (max_val - min_val)
            else:
                embeddings_matrix[:, i] = 0

        print(f"  Normalized embeddings (min-max scaling)")

        # Normalize rows for cosine similarity
        faiss.normalize_L2(embeddings_matrix)

        # Store embeddings back to Neo4j
        self._store_embeddings_to_neo4j(embeddings_matrix)

        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings_matrix)

        print(f"  FAISS index built with {self.faiss_index.ntotal} vectors")
        return True

    def _store_embeddings_to_neo4j(self, embeddings_matrix):
        """Store computed embeddings back to Neo4j"""
        print("\n Storing embeddings to Neo4j...")

        with self.driver.session() as session:
            for idx, embedding in enumerate(embeddings_matrix):
                hotel_id = self.idx_to_hotel_id[idx]
                session.run(
                    f"MATCH (h:Hotel {{hotel_id: $hotel_id}}) SET h.{self.EMBEDDING_PROP} = $embedding",
                    hotel_id=hotel_id,
                    embedding=embedding.tolist()
                )
        print(f"  Stored {len(embeddings_matrix)} embeddings")

    def verify_embeddings_exist(self):
        """Check if feature embeddings have been created for hotels"""
        print("\n Verifying feature embeddings...")

        query = f"""
        MATCH (h:Hotel)
        RETURN
            count(h) as total_hotels,
            count(h.{self.EMBEDDING_PROP}) as hotels_with_embeddings
        """

        with self.driver.session() as session:
            try:
                result = session.run(query)
                data = result.single()
                total = data['total_hotels']
                with_embeddings = data['hotels_with_embeddings']

                print(f"  Hotels with feature embeddings: {with_embeddings}/{total}")

                if with_embeddings == 0:
                    print("  No feature embeddings found!")
                    return False
                elif with_embeddings < total:
                    print(f"  Warning: {total - with_embeddings} hotels missing embeddings")
                else:
                    print("  All hotels have feature embeddings")

                return with_embeddings > 0

            except Exception as e:
                print(f"  Error verifying embeddings: {e}")
                return False

    def save_faiss_index(self):
        """Save FAISS index and mappings to disk"""
        print("\n Saving Feature FAISS index...")

        if self.faiss_index is None:
            print("  No FAISS index to save!")
            return False

        try:
            faiss.write_index(self.faiss_index, self.FAISS_INDEX_PATH)
            print(f"  Index saved to {self.FAISS_INDEX_PATH}")

            with open(self.HOTEL_MAPPING_PATH, 'wb') as f:
                pickle.dump({
                    'hotel_id_to_idx': self.hotel_id_to_idx,
                    'idx_to_hotel_id': self.idx_to_hotel_id
                }, f)
            print(f"  Mappings saved to {self.HOTEL_MAPPING_PATH}")

            return True
        except Exception as e:
            print(f"  Error saving: {e}")
            return False

    def load_faiss_index(self):
        """Load FAISS index and mappings from disk"""
        print("\n Loading Feature FAISS index...")

        if not os.path.exists(self.FAISS_INDEX_PATH):
            print(f"  Index file not found: {self.FAISS_INDEX_PATH}")
            return False

        if not os.path.exists(self.HOTEL_MAPPING_PATH):
            print(f"  Mapping file not found: {self.HOTEL_MAPPING_PATH}")
            return False

        try:
            self.faiss_index = faiss.read_index(self.FAISS_INDEX_PATH)
            print(f"  Index loaded ({self.faiss_index.ntotal} vectors)")

            with open(self.HOTEL_MAPPING_PATH, 'rb') as f:
                mappings = pickle.load(f)
            self.hotel_id_to_idx = mappings['hotel_id_to_idx']
            self.idx_to_hotel_id = mappings['idx_to_hotel_id']
            print(f"  Mappings loaded ({len(self.hotel_id_to_idx)} hotels)")

            return True
        except Exception as e:
            print(f"  Error loading: {e}")
            return False

    def find_similar_hotels(self, hotel_name, top_k=5):
        """Find similar hotels using feature embeddings"""
        print(f"\n [Feature] Searching similar hotels to: '{hotel_name}'")

        if self.faiss_index is None:
            print("  No FAISS index loaded. Attempting to load...")
            if not self.load_faiss_index():
                print("  Failed to load FAISS index. Run setup first.")
                return []

        query = """
        MATCH (h:Hotel {name: $hotel})
        RETURN h.hotel_id AS hotel_id
        """

        with self.driver.session() as session:
            result = session.run(query, hotel=hotel_name)
            record = result.single()

            if not record:
                print(f"  Hotel '{hotel_name}' not found in database")
                return []
            hotel_id = record['hotel_id']

        if hotel_id not in self.hotel_id_to_idx:
            print(f"  Hotel '{hotel_name}' does not have a feature embedding")
            return []

        idx = self.hotel_id_to_idx[hotel_id]
        query_vector = self.faiss_index.reconstruct(idx).reshape(1, -1)

        D, I = self.faiss_index.search(query_vector, top_k + 1)

        similar_hotels = []
        scores = []

        for i, score in zip(I[0], D[0]):
            result_hotel_id = self.idx_to_hotel_id[i]
            if result_hotel_id == hotel_id:
                continue
            similar_hotels.append(result_hotel_id)
            scores.append(score)
            if len(similar_hotels) >= top_k:
                break

        # Get hotel details
        query = """
        UNWIND $hotel_ids as hotel_id
        MATCH (h:Hotel {hotel_id: hotel_id})
        RETURN h.hotel_id as hotel_id, h.name as name,
               h.star_rating as star_rating, h.review_count as review_count,
               h.avg_reviewer_score as avg_score
        """
        with self.driver.session() as session:
            result = session.run(query, hotel_ids=similar_hotels)
            hotel_details = {record['hotel_id']: record for record in result.data()}

        final_results = []
        for hid, score in zip(similar_hotels, scores):
            if hid in hotel_details:
                details = hotel_details[hid]
                final_results.append({
                    'hotel': details['name'],
                    'hotel_id': hid,
                    'score': score,
                    'star_rating': details.get('star_rating'),
                    'review_count': details.get('review_count'),
                    'avg_score': details.get('avg_score')
                })
        return final_results

    def get_random_hotel(self):
        """Get a random hotel with feature embedding"""
        query = f"""
        MATCH (h:Hotel)
        WHERE h.{self.EMBEDDING_PROP} IS NOT NULL
        RETURN h.name as name, h.hotel_id as hotel_id
        ORDER BY rand()
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                return record['name'], record['hotel_id']
            return None, None

    def build_text_to_feature_mapper(self):
        """
        Build a mapper from text embeddings to feature space.
        Uses hotel descriptions (name + city + features) as training data.
        """
        print("\n Building text-to-feature mapper...")

        # Load text encoder
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Get hotel data with features
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(country:Country)
        RETURN h.hotel_id as hotel_id, h.name as name,
               c.name as city, country.name as country,
               h.star_rating as stars, h.review_count as reviews,
               h.avg_reviewer_score as avg_score,
               h.avg_cleanliness as cleanliness,
               h.avg_comfort as comfort,
               h.avg_facilities as facilities
        """

        with self.driver.session() as session:
            records = session.run(query).data()

        if not records:
            print("  No hotel data found!")
            return False

        # Create text descriptions for each hotel
        texts = []
        feature_vectors = []

        for record in records:
            hotel_id = record['hotel_id']
            if hotel_id not in self.hotel_id_to_idx:
                continue

            # Create descriptive text
            text = f"{record['name']} hotel in {record['city']}, {record['country']}. "
            text += f"{record['stars']:.0f} star hotel with {record['reviews']} reviews. "
            text += f"Average score {record['avg_score']:.1f}. "
            text += f"Cleanliness {record['cleanliness']:.1f}, comfort {record['comfort']:.1f}, facilities {record['facilities']:.1f}."
            texts.append(text)

            # Get corresponding feature embedding
            idx = self.hotel_id_to_idx[hotel_id]
            feature_vec = self.faiss_index.reconstruct(idx)
            feature_vectors.append(feature_vec)

        print(f"  Training on {len(texts)} hotel descriptions")

        # Encode texts
        text_embeddings = self.text_encoder.encode(texts, show_progress_bar=False)

        # Train Ridge regression mapper
        self.text_to_feature_mapper = Ridge(alpha=1.0)
        self.text_to_feature_mapper.fit(text_embeddings, np.array(feature_vectors))

        print("  Mapper trained successfully")
        return True

    # Synonym mappings for hotel name matching
    NAME_SYNONYMS = {
        'luxury': ['palace', 'grand', 'elite', 'royal', 'premium'],
        'cozy': ['inn', 'boutique', 'retreat', 'lodge', 'house'],
        'modern': ['elite', 'tower', 'heights'],
        'traditional': ['grand', 'palace', 'royal'],
        'beach': ['harbour', 'harbor', 'oasis', 'bay'],
        'city': ['tower', 'heights', 'elite', 'mitte'],
    }

    # Attribute keywords mapped to feature properties
    ATTRIBUTE_KEYWORDS = {
        'clean': 'avg_cleanliness',
        'cleanliness': 'avg_cleanliness',
        'spotless': 'avg_cleanliness',
        'comfortable': 'avg_comfort',
        'comfort': 'avg_comfort',
        'cozy': 'avg_comfort',
        'facilities': 'avg_facilities',
        'amenities': 'avg_facilities',
        'equipped': 'avg_facilities',
        'location': 'avg_location',
        'central': 'avg_location',
        'staff': 'avg_staff',
        'service': 'avg_staff',
        'friendly': 'avg_staff',
        'value': 'avg_value',
        'affordable': 'avg_value',
        'worth': 'avg_value',
    }

    # Star rating keywords
    STAR_KEYWORDS = {
        'luxury': 5,
        'luxurious': 5,
        'premium': 5,
        'five star': 5,
        '5 star': 5,
        '5-star': 5,
        'high-end': 5,
        'upscale': 5,
        'mid-range': 4,
        'four star': 4,
        '4 star': 4,
        '4-star': 4,
        'budget': 3,
        'cheap': 3,
        'affordable': 3,
        'three star': 3,
        '3 star': 3,
        '3-star': 3,
    }

    # Ranking trigger words - when these appear, use attribute-based sorting
    RANKING_TRIGGERS = ['best', 'top', 'highest', 'most', 'greatest', 'cleanest', 'nicest']

    # Map attribute keywords to database property names for ranking queries
    RANKING_ATTRIBUTES = {
        'clean': 'avg_cleanliness',
        'cleanliness': 'avg_cleanliness',
        'cleanest': 'avg_cleanliness',
        'comfortable': 'avg_comfort',
        'comfort': 'avg_comfort',
        'facilities': 'avg_facilities',
        'amenities': 'avg_facilities',
        'location': 'avg_location',
        'located': 'avg_location',
        'staff': 'avg_staff',
        'service': 'avg_staff',
        'value': 'avg_value',
        'rating': 'avg_reviewer_score',
        'rated': 'avg_reviewer_score',
        'score': 'avg_reviewer_score',
        'reviewed': 'review_count',
        'reviews': 'review_count',
        'popular': 'review_count',
    }

    def _get_ranking_attribute(self, query_text):
        """
        Detect if query is asking for ranking by a specific attribute.
        Returns (attribute_name, db_property) or (None, None) if not a ranking query.

        Examples:
        - "best hotels in cleanliness" -> ('cleanliness', 'avg_cleanliness')
        - "top rated hotels" -> ('rated', 'avg_reviewer_score')
        - "most popular hotels" -> ('popular', 'review_count')
        """
        query_lower = query_text.lower()
        words = query_lower.split()

        # Check if any ranking trigger word is present
        has_ranking_trigger = any(trigger in query_lower for trigger in self.RANKING_TRIGGERS)

        if not has_ranking_trigger:
            return (None, None)

        # Find which attribute is being ranked
        for attr_word, db_prop in self.RANKING_ATTRIBUTES.items():
            if attr_word in words or attr_word in query_lower:
                return (attr_word, db_prop)

        return (None, None)

    def _get_location_filter(self, query_text):
        """
        Extract city/country names from query for location filtering.
        Returns (city, country) tuple or (None, None) if no location found.
        """
        query_lower = query_text.lower()

        # Get all cities and countries from database
        with self.driver.session() as session:
            cities = session.run("MATCH (c:City) RETURN c.name as name").data()
            countries = session.run("MATCH (c:Country) RETURN c.name as name").data()

        city_names = {c['name'].lower(): c['name'] for c in cities}
        country_names = {c['name'].lower(): c['name'] for c in countries}

        found_city = None
        found_country = None

        # Check for city matches
        for city_lower, city_original in city_names.items():
            if city_lower in query_lower:
                found_city = city_original
                break

        # Check for country matches
        for country_lower, country_original in country_names.items():
            if country_lower in query_lower:
                found_country = country_original
                break

        return found_city, found_country

    def _get_star_rating_filter(self, query_text):
        """
        Extract star rating preference from query.
        Returns (min_stars, max_stars) tuple or (None, None) if not specified.
        """
        query_lower = query_text.lower()

        # Check for star rating keywords (check multi-word first)
        for keyword, stars in sorted(self.STAR_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if keyword in query_lower:
                if stars == 5:
                    return (5, 5)  # Luxury = exactly 5 stars
                elif stars == 4:
                    return (4, 5)  # Mid-range = 4-5 stars
                elif stars == 3:
                    return (3, 4)  # Budget = 3-4 stars
        return (None, None)

    def _get_attribute_boost(self, query_text, hotel_details):
        """
        Calculate boost based on attribute keywords in query.
        If query mentions 'clean', boost hotels with high cleanliness scores.
        Returns a multiplier (1.0 = no boost).
        """
        query_lower = query_text.lower()
        boost = 1.0

        for keyword, attribute in self.ATTRIBUTE_KEYWORDS.items():
            if keyword in query_lower:
                # Get the attribute value - try both with and without 'avg_' prefix
                attr_value = hotel_details.get(attribute, 0)
                if attr_value is None:
                    # Try without avg_ prefix (for backwards compatibility)
                    attr_value = hotel_details.get(attribute.replace('avg_', ''), 0)
                if attr_value is None:
                    attr_value = 0

                # Boost based on how high the attribute is (scale 0-10)
                # Hotels with score >= 9 get 15% boost, >= 8 get 10% boost
                if attr_value >= 9.0:
                    boost *= 1.15
                elif attr_value >= 8.5:
                    boost *= 1.10
                elif attr_value >= 8.0:
                    boost *= 1.05

        return boost

    def _fuzzy_match(self, word1, word2, threshold=0.85):
        """
        Fuzzy matching for typo detection.
        Returns True if words are similar enough (likely a typo).
        """
        if len(word1) < 4 or len(word2) < 4:
            return word1 == word2

        # Check if one is a near-complete substring of the other
        # (handles cases like "azur" in "azure")
        if len(word1) >= 4 and len(word2) >= 4:
            if word1 in word2 and len(word1) >= len(word2) - 2:
                return True
            if word2 in word1 and len(word2) >= len(word1) - 2:
                return True

        # Require similar length (within 2 characters)
        if abs(len(word1) - len(word2)) > 2:
            return False

        # Check prefix match (first 3 characters must match for typo detection)
        if word1[:3] != word2[:3]:
            return False

        # Character-based similarity with stricter threshold
        set1, set2 = set(word1), set(word2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        similarity = intersection / union if union > 0 else 0

        return similarity >= threshold

    def _get_name_match_boost(self, query_text, hotel_name):
        """
        Calculate a boost score if query words match hotel name.
        Includes:
        - Direct word matching (20% per word)
        - Synonym matching (15% per synonym)
        - Multi-word phrase matching (extra 15% bonus)
        - Fuzzy matching for typos (10% per fuzzy match)
        Returns a boost multiplier (1.0 = no boost, higher = boost).
        """
        query_lower = query_text.lower()
        query_words = set(query_lower.split())
        hotel_name_lower = hotel_name.lower().replace("'", "")
        hotel_words = set(hotel_name_lower.split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'in', 'at', 'for', 'with', 'and', 'or', 'hotel', 'hotels', 'good', 'best', 'top', 'great', 'nice'}
        query_words_filtered = query_words - stop_words

        boost = 1.0
        match_reasons = []

        # 1. Direct word matching (20% per word)
        direct_matches = query_words_filtered & hotel_words
        if direct_matches:
            boost += 0.2 * len(direct_matches)
            match_reasons.append(f"direct:{','.join(direct_matches)}")

        # 2. Multi-word phrase matching (extra 15% if 2+ consecutive words match)
        for i in range(len(query_lower.split()) - 1):
            phrase = ' '.join(query_lower.split()[i:i+2])
            if phrase in hotel_name_lower:
                boost += 0.15
                match_reasons.append(f"phrase:{phrase}")
                break  # Only count once

        # 3. Synonym matching (15% per synonym match)
        for query_word in query_words_filtered:
            if query_word in self.NAME_SYNONYMS:
                synonyms = self.NAME_SYNONYMS[query_word]
                for synonym in synonyms:
                    if synonym in hotel_words:
                        boost += 0.15
                        match_reasons.append(f"synonym:{query_word}->{synonym}")
                        break  # Only count once per query word

        # 4. Fuzzy matching for typos (15% per fuzzy match)
        if not direct_matches:  # Only use fuzzy if no direct matches
            for query_word in query_words_filtered:
                if len(query_word) >= 4:  # Only fuzzy match longer words
                    for hotel_word in hotel_words:
                        if self._fuzzy_match(query_word, hotel_word):
                            boost += 0.15
                            match_reasons.append(f"fuzzy:{query_word}~{hotel_word}")
                            break

        if boost > 1.0:
            return (boost, match_reasons)
        return (1.0, [])

    def search_by_query(self, query_text, top_k=5):
        """
        Search for hotels using a natural language query.
        Features:
        - Ranking queries (best/top in X -> sort by attribute)
        - Location filtering (city/country extraction)
        - Star rating filtering (luxury/budget keywords)
        - Name matching with synonyms and fuzzy matching
        - Attribute boosting (clean, comfortable, etc.)
        - Feature similarity ranking
        """
        print(f"\n [Feature] Searching by query: '{query_text}'")

        # Ensure FAISS index is loaded
        if self.faiss_index is None:
            print("  No FAISS index loaded. Attempting to load...")
            if not self.load_faiss_index():
                print("  Failed to load FAISS index. Run setup first.")
                return []

        # Build mapper if not exists
        if not hasattr(self, 'text_to_feature_mapper') or self.text_to_feature_mapper is None:
            print("  Building text-to-feature mapper...")
            if not self.build_text_to_feature_mapper():
                print("  Failed to build mapper")
                return []

        # Check for ranking query (e.g., "best hotels in cleanliness")
        ranking_attr, ranking_prop = self._get_ranking_attribute(query_text)

        # Extract filters from query
        filter_city, filter_country = self._get_location_filter(query_text)
        min_stars, max_stars = self._get_star_rating_filter(query_text)

        # Print active filters
        if ranking_attr:
            print(f"  Ranking by: {ranking_attr} ({ranking_prop})")
        if filter_city:
            print(f"  Location filter: city = '{filter_city}'")
        elif filter_country:
            print(f"  Location filter: country = '{filter_country}'")
        if min_stars:
            print(f"  Star rating filter: {min_stars}-{max_stars} stars")

        # Encode query
        query_embedding = self.text_encoder.encode([query_text], show_progress_bar=False)

        # Map to feature space
        query_feature = self.text_to_feature_mapper.predict(query_embedding)
        query_feature = query_feature.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_feature)

        # Search FAISS - get all results to allow name matching and re-ranking
        # We search all hotels and then apply filters + boosts before limiting to top_k
        D, I = self.faiss_index.search(query_feature, self.faiss_index.ntotal)

        # Get hotel details
        hotel_ids = [self.idx_to_hotel_id[i] for i in I[0]]

        query = """
        UNWIND $hotel_ids as hotel_id
        MATCH (h:Hotel {hotel_id: hotel_id})-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(country:Country)
        RETURN h.hotel_id as hotel_id, h.name as name,
               c.name as city, country.name as country,
               h.star_rating as star_rating, h.review_count as review_count,
               h.avg_reviewer_score as avg_score,
               h.avg_cleanliness as avg_cleanliness, h.avg_comfort as avg_comfort,
               h.avg_facilities as avg_facilities, h.avg_location as avg_location,
               h.avg_staff as avg_staff, h.avg_value as avg_value
        """

        with self.driver.session() as session:
            result = session.run(query, hotel_ids=hotel_ids)
            hotel_details = {r['hotel_id']: r for r in result.data()}

        results = []
        for idx, (i, score) in enumerate(zip(I[0], D[0])):
            hotel_id = self.idx_to_hotel_id[i]
            if hotel_id in hotel_details:
                details = hotel_details[hotel_id]

                # Apply location filter
                if filter_city and details['city'] != filter_city:
                    continue
                if filter_country and details['country'] != filter_country:
                    continue

                # Apply star rating filter
                star_rating = details.get('star_rating', 0)
                if min_stars and (star_rating < min_stars or star_rating > max_stars):
                    continue

                # Apply name match boost (now returns tuple)
                name_boost, match_reasons = self._get_name_match_boost(query_text, details['name'])

                # Apply attribute boost
                attr_boost = self._get_attribute_boost(query_text, details)

                # Combine boosts
                total_boost = name_boost * attr_boost
                boosted_score = float(score) * total_boost

                # Get ranking value if this is a ranking query
                ranking_value = None
                if ranking_prop:
                    ranking_value = details.get(ranking_prop, 0) or 0

                results.append({
                    'hotel': details['name'],
                    'hotel_id': hotel_id,
                    'city': details['city'],
                    'country': details['country'],
                    'score': boosted_score,
                    'base_score': float(score),
                    'name_boost': name_boost,
                    'attr_boost': attr_boost,
                    'match_reasons': match_reasons,
                    'ranking_value': ranking_value,
                    'ranking_attr': ranking_attr,
                    'star_rating': details.get('star_rating'),
                    'review_count': details.get('review_count'),
                    'avg_score': details.get('avg_score'),
                    'cleanliness': details.get('avg_cleanliness'),
                    'comfort': details.get('avg_comfort'),
                    'facilities': details.get('avg_facilities'),
                    'location': details.get('avg_location'),
                    'staff': details.get('avg_staff'),
                    'value': details.get('avg_value')
                })

        # Re-sort results
        if ranking_prop:
            # Ranking query: sort by the attribute value (e.g., "best in cleanliness" -> sort by cleanliness)
            results.sort(key=lambda x: (x['ranking_value'] or 0), reverse=True)
        else:
            # Regular query: sort by boosted similarity score
            results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def setup_embeddings(self):
        """Complete Feature Embedding setup"""
        print("\n" + "="*60)
        print("SETTING UP FEATURE-BASED EMBEDDINGS")
        print("="*60)
        print("Using enriched hotel properties:")
        print(f"  {self.FEATURE_PROPERTIES}")
        print("="*60)

        try:
            # Step 1: Enrich hotel nodes with computed properties
            if not self.enrich_hotel_properties():
                print("\n Setup failed: Could not enrich hotel properties")
                return False

            # Step 2: Build feature embeddings
            if not self.build_feature_embeddings():
                print("\n Setup failed: Could not build feature embeddings")
                return False

            # Step 3: Verify embeddings
            if not self.verify_embeddings_exist():
                print("\n Setup failed: No embeddings created")
                return False

            # Step 4: Save FAISS index
            if not self.save_faiss_index():
                print("\n Warning: Could not save FAISS index")

            print("\n" + "="*60)
            print(" FEATURE EMBEDDING SETUP COMPLETE")
            print("="*60)
            return True

        except Exception as e:
            print(f"\nSetup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# ==============================================================================
# UNIFIED RETRIEVER - BLACK BOX INTERFACE
# ==============================================================================

class HotelEmbeddingRetriever:
    """
    Unified Hotel Embedding Retriever - Black Box Interface

    This class provides a single interface for hotel similarity search,
    abstracting away the underlying embedding method (Node2Vec or Feature-based).

    Usage:
        # Create retriever with desired model
        retriever = HotelEmbeddingRetriever(driver, model_type='feature')

        # Search - works the same regardless of model
        results = retriever.search_by_query("luxury hotel in Paris", top_k=5)
        results = retriever.find_similar_hotels("The Azure Tower", top_k=5)

    Model types:
        - 'node2vec': Graph structure-based embeddings (128-dim)
                      Finds hotels similar based on how they connect via reviews/travelers
        - 'feature': Property-based embeddings (12-dim)
                     Finds hotels similar based on ratings, scores, and computed review data
                     Supports natural language query search with location/attribute filtering
    """

    VALID_MODELS = ['node2vec', 'feature']

    def __init__(self, driver, model_type='feature', auto_initialize=True):
        """
        Initialize the unified retriever.

        Args:
            driver: Neo4j database driver
            model_type: 'node2vec' or 'feature' (default: 'feature')
            auto_initialize: If True, automatically loads existing index or sets up if needed
        """
        if model_type not in self.VALID_MODELS:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be one of: {self.VALID_MODELS}")

        self.driver = driver
        self._model_type = model_type
        self._retriever = None
        self._initialized = False

        # Create the appropriate retriever
        self._create_retriever()

        # Auto-initialize if requested
        if auto_initialize:
            self._auto_initialize()

    @property
    def model_type(self):
        """Get current model type"""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        """
        Change the model type. This will switch to the new retriever.
        Note: You may need to call initialize() after switching.
        """
        if value not in self.VALID_MODELS:
            raise ValueError(f"Invalid model_type '{value}'. Must be one of: {self.VALID_MODELS}")

        if value != self._model_type:
            self._model_type = value
            self._retriever = None
            self._initialized = False
            self._create_retriever()
            self._auto_initialize()

    def _create_retriever(self):
        """Create the internal retriever based on model type"""
        if self._model_type == 'node2vec':
            self._retriever = EmbeddingRetriever(self.driver)
        else:  # 'feature'
            self._retriever = FeatureEmbeddingRetriever(self.driver)
        print(f"  Retriever created: {self._model_type}")

    def _auto_initialize(self):
        """
        Automatically initialize the retriever:
        1. Try to load existing FAISS index
        2. If not found, run full setup
        """
        print(f"\n Initializing {self._model_type} retriever...")

        # Try to load existing index
        if self._retriever.load_faiss_index():
            print(f"  Loaded existing {self._model_type} index")
            self._initialized = True

            # For feature model, also build the text mapper if needed
            if self._model_type == 'feature':
                if not hasattr(self._retriever, 'text_to_feature_mapper') or self._retriever.text_to_feature_mapper is None:
                    self._retriever.build_text_to_feature_mapper()
            return True

        # No existing index - check if embeddings exist in Neo4j
        if self._retriever.verify_embeddings_exist():
            # Embeddings exist, just build FAISS index
            print(f"  Embeddings exist, building FAISS index...")
            if self._retriever.build_faiss_index() if self._model_type == 'node2vec' else self._retriever.build_feature_embeddings():
                self._retriever.save_faiss_index()
                self._initialized = True

                if self._model_type == 'feature':
                    self._retriever.build_text_to_feature_mapper()
                return True

        # No embeddings - need full setup
        print(f"  No existing embeddings found. Run setup_embeddings() to create them.")
        self._initialized = False
        return False

    def setup_embeddings(self):
        """
        Run full embedding setup for the current model type.
        This creates embeddings from scratch and builds the FAISS index.
        """
        success = self._retriever.setup_embeddings()
        if success:
            self._initialized = True

            # For feature model, build text mapper
            if self._model_type == 'feature':
                self._retriever.build_text_to_feature_mapper()

        return success

    def is_initialized(self):
        """Check if the retriever is ready for queries"""
        return self._initialized and self._retriever.faiss_index is not None

    def find_similar_hotels(self, hotel_name, top_k=5):
        """
        Find hotels similar to the given hotel.

        Works with both model types:
        - node2vec: Finds structurally similar hotels (same traveler patterns)
        - feature: Finds hotels with similar properties (ratings, reviews)

        Args:
            hotel_name: Name of the reference hotel
            top_k: Number of similar hotels to return

        Returns:
            List of dicts with hotel details and similarity scores
        """
        if not self.is_initialized():
            print("  Retriever not initialized. Call setup_embeddings() first.")
            return []

        return self._retriever.find_similar_hotels(hotel_name, top_k=top_k)

    def search_by_query(self, query_text, top_k=5):
        """
        Search for hotels using natural language query.

        Features (available in both models, more advanced in 'feature'):
        - Location filtering (city/country extraction)
        - Star rating filtering (luxury/budget keywords)
        - Name matching with synonyms and fuzzy matching
        - Attribute boosting (clean, comfortable, etc.)

        Args:
            query_text: Natural language search query
            top_k: Number of results to return

        Returns:
            List of dicts with hotel details and relevance scores
        """
        if not self.is_initialized():
            print("  Retriever not initialized. Call setup_embeddings() first.")
            return []

        # Feature model has native query search
        if self._model_type == 'feature':
            return self._retriever.search_by_query(query_text, top_k=top_k)

        # Node2Vec doesn't have query search - use fallback approach
        # Try to find a hotel name in the query and use find_similar_hotels
        print(f"  Note: query search with node2vec uses basic keyword matching")
        return self._search_by_query_fallback(query_text, top_k)

    def _search_by_query_fallback(self, query_text, top_k):
        """
        Fallback query search for Node2Vec model.
        Performs basic keyword matching against hotel names and properties.
        """
        query_lower = query_text.lower()

        # Get all hotels
        query = """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(country:Country)
        RETURN h.hotel_id as hotel_id, h.name as name,
               c.name as city, country.name as country,
               h.star_rating as star_rating
        """

        with self.driver.session() as session:
            records = session.run(query).data()

        # Score hotels based on keyword matches
        results = []
        for record in records:
            score = 0.0

            # Location match
            if record['city'].lower() in query_lower:
                score += 0.5
            if record['country'].lower() in query_lower:
                score += 0.3

            # Name match
            name_lower = record['name'].lower()
            for word in query_lower.split():
                if len(word) > 3 and word in name_lower:
                    score += 0.2

            if score > 0:
                results.append({
                    'hotel': record['name'],
                    'hotel_id': record['hotel_id'],
                    'city': record['city'],
                    'country': record['country'],
                    'score': score,
                    'star_rating': record.get('star_rating')
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def get_random_hotel(self):
        """Get a random hotel from the database"""
        return self._retriever.get_random_hotel()

    def verify_embeddings_exist(self):
        """Check if embeddings have been created"""
        return self._retriever.verify_embeddings_exist()

    def get_model_info(self):
        """Get information about the current model"""
        if self._model_type == 'node2vec':
            return {
                'model_type': 'node2vec',
                'description': 'Graph structure-based embeddings',
                'embedding_dim': 128,
                'features': ['find_similar_hotels', 'basic_query_search'],
                'best_for': 'Finding hotels with similar traveler patterns'
            }
        else:
            return {
                'model_type': 'feature',
                'description': 'Property-based embeddings with enriched hotel data',
                'embedding_dim': 12,
                'features': ['find_similar_hotels', 'advanced_query_search',
                            'location_filtering', 'star_rating_filtering',
                            'attribute_boosting', 'synonym_matching', 'fuzzy_matching'],
                'best_for': 'Natural language hotel search with filtering'
            }


# ==============================================================================
# SIMPLE FUNCTION INTERFACE FOR EXTERNAL SCRIPTS
# ==============================================================================

# Global retriever instance (lazy loaded)
_global_retriever = None
_global_model_type = 'feature'  # Default model


def set_model_type(model_type):
    """
    Set the global model type for subsequent searches.

    Args:
        model_type: 'node2vec' or 'feature'
    """
    global _global_model_type, _global_retriever
    if model_type not in ['node2vec', 'feature']:
        raise ValueError(f"Invalid model_type. Must be 'node2vec' or 'feature'")

    if model_type != _global_model_type:
        _global_model_type = model_type
        _global_retriever = None  # Force re-initialization


def get_model_type():
    """Get the current global model type"""
    return _global_model_type


def _get_retriever():
    """Get or create the global retriever instance"""
    global _global_retriever
    if _global_retriever is None:
        db_manager = DBManager()
        _global_retriever = HotelEmbeddingRetriever(
            db_manager.driver,
            model_type=_global_model_type,
            auto_initialize=True
        )
    return _global_retriever


def search_hotels(query, top_k=5):
    """
    Search for hotels using natural language query.

    This is the main entry point for external scripts.
    Uses the globally configured model type.

    Args:
        query: Natural language search query (e.g., "luxury hotel in Paris")
        top_k: Number of results to return

    Returns:
        List of hotel results with scores

    Example:
        from embeddings_retreiver import search_hotels, set_model_type

        # Use feature-based search (default)
        results = search_hotels("clean comfortable hotel in Tokyo")

        # Switch to node2vec
        set_model_type('node2vec')
        results = search_hotels("hotel near beach")
    """
    retriever = _get_retriever()
    return retriever.search_by_query(query, top_k=top_k)


def find_similar(hotel_name, top_k=5):
    """
    Find hotels similar to the given hotel.

    Args:
        hotel_name: Name of the reference hotel
        top_k: Number of similar hotels to return

    Returns:
        List of similar hotels with scores

    Example:
        from embeddings_retreiver import find_similar

        results = find_similar("The Azure Tower", top_k=10)
    """
    retriever = _get_retriever()
    return retriever.find_similar_hotels(hotel_name, top_k=top_k)


def setup_embeddings():
    """
    Run embedding setup for the current model type.
    Call this if embeddings haven't been created yet.

    Returns:
        True if setup succeeded, False otherwise
    """
    retriever = _get_retriever()
    return retriever.setup_embeddings()


def get_retriever(model_type='feature', driver=None):
    """
    Get a retriever instance for custom usage.

    Args:
        model_type: 'node2vec' or 'feature'
        driver: Neo4j driver (optional, will create one if not provided)

    Returns:
        HotelEmbeddingRetriever instance

    Example:
        from embeddings_retreiver import get_retriever

        retriever = get_retriever('feature')
        results = retriever.search_by_query("luxury hotel")
    """
    if driver is None:
        db_manager = DBManager()
        driver = db_manager.driver

    return HotelEmbeddingRetriever(driver, model_type=model_type, auto_initialize=True)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Hotel Similarity using Graph Embeddings')
    parser.add_argument('--method', type=str, default='feature', choices=['node2vec', 'feature'],
                        help='Embedding method: node2vec (graph structure) or feature (enriched properties)')
    parser.add_argument('--setup', action='store_true', help='Setup embeddings')
    parser.add_argument('--hotel', type=str, help='Hotel name to find similar hotels for')
    parser.add_argument('--query', type=str, help='Natural language query to search hotels')
    parser.add_argument('--top-k', type=int, default=5, help='Number of similar hotels to return')
    parser.add_argument('--random', action='store_true', help='Find similar hotels for a random hotel')
    parser.add_argument('--compare', action='store_true', help='Compare Node2Vec vs Feature embedding results')

    args = parser.parse_args()

    try:
        print("Connecting to Neo4j...")
        db_manager = DBManager()

        # Use unified retriever (black box interface)
        retriever = HotelEmbeddingRetriever(
            db_manager.driver,
            model_type=args.method,
            auto_initialize=not args.setup  # Don't auto-init if we're doing setup
        )
        info = retriever.get_model_info()
        print(f"  Using: {info['model_type']} ({info['description']})")

        # Setup embeddings
        if args.setup:
            success = retriever.setup_embeddings()
            if not success:
                db_manager.close()
                exit(1)

        # Find similar hotels
        if args.hotel or args.random:
            if not retriever.verify_embeddings_exist():
                print(f"\n  No embeddings found. Run with --setup --method {args.method} first")
                db_manager.close()
                exit(1)

            if args.random:
                hotel_name, hotel_id = retriever.get_random_hotel()
                if not hotel_name:
                    print(" No hotels with embeddings found")
                    db_manager.close()
                    exit(1)
                print(f"\n Random hotel selected: {hotel_name} (ID: {hotel_id})")
            else:
                hotel_name = args.hotel

            similar_hotels = retriever.find_similar_hotels(hotel_name, top_k=args.top_k)

            if similar_hotels:
                print(f"\n Similar Hotels ({args.method.upper()}):")
                print("-" * 60)
                for i, record in enumerate(similar_hotels, 1):
                    print(f"{i}. {record['hotel']}")
                    print(f"   Score: {record['score']:.4f}")
                    if 'star_rating' in record and record['star_rating']:
                        print(f"   Stars: {record['star_rating']} | Reviews: {record.get('review_count', 'N/A')} | Avg Score: {record.get('avg_score', 'N/A'):.1f}")
                    print()
            else:
                print(f" No similar hotels found for '{hotel_name}'")

        # Search by query (unified interface - works with both models)
        if args.query:
            results = retriever.search_by_query(args.query, top_k=args.top_k)

            if results:
                print(f"\n Search Results for: '{args.query}'")
                print("-" * 70)
                for i, r in enumerate(results, 1):
                    # Build boost indicators (only available in feature model)
                    boosts = []
                    if r.get('match_reasons'):
                        boosts.extend([str(reason) for reason in r['match_reasons']])
                    if r.get('attr_boost', 1.0) > 1.0:
                        boosts.append(f"attr:{r['attr_boost']:.2f}x")

                    boost_str = f" [{', '.join(boosts)}]" if boosts else ""
                    city = r.get('city', 'N/A')
                    country = r.get('country', 'N/A')
                    print(f"{i}. {r['hotel']} ({city}, {country}){boost_str}")
                    print(f"   Score: {r['score']:.4f} | Stars: {r.get('star_rating', 'N/A')}")

                    # Extra details for feature model
                    if r.get('avg_score') is not None:
                        print(f"   Avg Score: {r['avg_score']:.1f} | Clean: {r.get('cleanliness', 0):.1f} | Comfort: {r.get('comfort', 0):.1f}")
                    print()
            else:
                print(f" No results found for query: '{args.query}'")

        # Compare both methods
        if args.compare:
            print("\n" + "="*70)
            print("COMPARING NODE2VEC vs FEATURE EMBEDDINGS")
            print("="*70)

            # Use unified retrievers for both methods
            n2v = HotelEmbeddingRetriever(db_manager.driver, model_type='node2vec', auto_initialize=True)
            feat = HotelEmbeddingRetriever(db_manager.driver, model_type='feature', auto_initialize=True)

            hotel_name = None
            if feat.is_initialized():
                hotel_name, _ = feat.get_random_hotel()
            if not hotel_name and n2v.is_initialized():
                hotel_name, _ = n2v.get_random_hotel()

            if hotel_name:
                print(f"\nQuery Hotel: {hotel_name}")
                top_k = args.top_k

                print(f"\n--- NODE2VEC Results (graph structure) ---")
                if n2v.is_initialized():
                    results = n2v.find_similar_hotels(hotel_name, top_k=top_k)
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. {r['hotel']} (score: {r['score']:.4f})")
                else:
                    print("  Run --setup --method node2vec first")

                print(f"\n--- FEATURE Results (enriched properties) ---")
                if feat.is_initialized():
                    results = feat.find_similar_hotels(hotel_name, top_k=top_k)
                    for i, r in enumerate(results, 1):
                        print(f"  {i}. {r['hotel']} (score: {r['score']:.4f})")
                        if r.get('star_rating'):
                            print(f"      Stars: {r['star_rating']} | Reviews: {r.get('review_count', 'N/A')}")
                else:
                    print("  Run --setup --method feature first")

                print("\n" + "="*70)
            else:
                print("No embeddings found. Run --setup first.")

        elif not args.setup and not args.hotel and not args.random and not args.query:
            parser.print_help()
            print("\n" + "="*70)
            print("USAGE EXAMPLES")
            print("="*70)
            print("\n  CLI Usage:")
            print("  -----------")
            print("  # Setup Feature Embedding (default, recommended)")
            print("  python embeddings_retreiver.py --setup")
            print()
            print("  # Setup Node2Vec (graph structure)")
            print("  python embeddings_retreiver.py --setup --method node2vec")
            print()
            print("  # Search by natural language query")
            print("  python embeddings_retreiver.py --query 'luxury hotel in Paris'")
            print()
            print("  # Find similar hotels")
            print("  python embeddings_retreiver.py --hotel 'The Azure Tower'")
            print()
            print("  # Compare both methods")
            print("  python embeddings_retreiver.py --compare")
            print()
            print("\n  External Script Usage (Black Box Interface):")
            print("  -----------------------------------------------")
            print("  # Simple function interface:")
            print("  from embeddings_retreiver import search_hotels, find_similar, set_model_type")
            print()
            print("  results = search_hotels('luxury hotel in Paris')  # Uses feature model by default")
            print("  results = find_similar('The Azure Tower')")
            print()
            print("  set_model_type('node2vec')  # Switch to node2vec model")
            print("  results = search_hotels('beach hotel')")
            print()
            print("\n  # Class interface (more control):")
            print("  from embeddings_retreiver import HotelEmbeddingRetriever")
            print("  from db_manager import DBManager")
            print()
            print("  db = DBManager()")
            print("  retriever = HotelEmbeddingRetriever(db.driver, model_type='feature')")
            print("  results = retriever.search_by_query('clean comfortable hotel')")
            print()
            print("  # Switch model at runtime:")
            print("  retriever.model_type = 'node2vec'")
            print("  results = retriever.find_similar_hotels('Grand Hotel')")
            print("="*70)

        db_manager.close()

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
