
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
# FASTRP EMBEDDING RETRIEVER (Second Node Embedding Model)
# ==============================================================================

class FastRPEmbeddingRetriever:
    """
    FastRP (Fast Random Projection) Node Embedding Retriever.

    A second graph embedding technique for comparison with Node2Vec.
    FastRP uses random projections and is generally faster than Node2Vec.

    Key differences from Node2Vec:
    - Algorithm: Random projection vs random walks
    - Speed: Faster computation
    - Parameters: iterationWeights vs walkLength/iterations
    """

    GRAPH_NAME = "travelGraph"
    EMBEDDING_PROP = "fastRPEmbedding"
    EMBEDDING_DIM = 128
    VECTOR_INDEX = "hotel_fastrp_index"

    FAISS_INDEX_PATH = "faiss_hotel_fastrp_index.bin"
    HOTEL_MAPPING_PATH = "hotel_mapping_fastrp.pkl"

    def __init__(self, driver):
        self.driver = driver
        self.faiss_index = None
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}

    def create_graph(self):
        """Create GDS graph projection (same as Node2Vec)"""
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
            try:
                session.run(drop_query)
                print("  Dropped existing graph projection")
            except ClientError as e:
                if "not found" not in str(e).lower():
                    print(f"Warning: Could not drop graph: {e}")

            try:
                result = session.run(create_query)
                data = result.data()[0]
                print(f"  Graph created: {data['nodeCount']} nodes, {data['relationshipCount']} relationships")
                return data
            except ClientError as e:
                print(f" Error creating graph: {e}")
                raise

    def run_fastrp(self):
        """Run FastRP algorithm to generate embeddings"""
        print("\n Running FastRP algorithm...")

        query = f"""
        CALL gds.fastRP.write(
          '{self.GRAPH_NAME}',
          {{
            writeProperty: '{self.EMBEDDING_PROP}',
            embeddingDimension: {self.EMBEDDING_DIM},
            iterationWeights: [0.0, 1.0, 1.0, 1.0],
            randomSeed: 42
          }}
        )
        """

        with self.driver.session() as session:
            try:
                result = session.run(query)
                data = result.data()[0]
                print(f"  FastRP complete:")
                print(f"    - Nodes processed: {data.get('nodePropertiesWritten', 'N/A')}")
                print(f"    - Computation time: {data.get('computeMillis', 0)/1000:.2f}s")
                return data
            except ClientError as e:
                print(f" Error running FastRP: {e}")
                raise

    def create_vector_index(self):
        """Create vector index for similarity search"""
        print("\n Creating FastRP vector index...")

        drop_query = f"DROP INDEX {self.VECTOR_INDEX} IF EXISTS"
        create_query = f"""
        CREATE VECTOR INDEX {self.VECTOR_INDEX} IF NOT EXISTS
        FOR (h:Hotel)
        ON (h.{self.EMBEDDING_PROP})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {self.EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """

        with self.driver.session() as session:
            try:
                session.run(drop_query)
                session.run(create_query)
                print(f"  Vector index '{self.VECTOR_INDEX}' created")
            except ClientError as e:
                print(f"  Error creating vector index: {e}")
                raise

    def verify_embeddings_exist(self):
        """Check if FastRP embeddings exist for hotels"""
        print("\n Verifying FastRP embeddings...")

        query = f"""
        MATCH (h:Hotel)
        RETURN
            count(h) as total_hotels,
            count(h.{self.EMBEDDING_PROP}) as hotels_with_embeddings
        """

        with self.driver.session() as session:
            result = session.run(query)
            data = result.single()
            total = data['total_hotels']
            with_emb = data['hotels_with_embeddings']

            if with_emb > 0:
                print(f"  Found {with_emb}/{total} hotels with FastRP embeddings")
                return True
            else:
                print(f"  No FastRP embeddings found. Run setup_embeddings() first.")
                return False

    def build_faiss_index(self):
        """Build FAISS index from FastRP embeddings"""
        print("\n Building FAISS index from FastRP embeddings...")

        query = f"""
        MATCH (h:Hotel)
        WHERE h.{self.EMBEDDING_PROP} IS NOT NULL
        RETURN h.hotel_id as hotel_id, h.{self.EMBEDDING_PROP} as embedding
        ORDER BY h.hotel_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)

        if not records:
            print("  No embeddings found!")
            return False

        embeddings = []
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}

        for idx, record in enumerate(records):
            hotel_id = record['hotel_id']
            embedding = np.array(record['embedding'], dtype='float32')
            embeddings.append(embedding)
            self.hotel_id_to_idx[hotel_id] = idx
            self.idx_to_hotel_id[idx] = hotel_id

        embeddings_matrix = np.vstack(embeddings)
        faiss.normalize_L2(embeddings_matrix)

        self.faiss_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.faiss_index.add(embeddings_matrix)

        print(f"  FAISS index built with {self.faiss_index.ntotal} vectors")
        return True

    def save_faiss_index(self):
        """Save FAISS index and mappings to disk"""
        print("\n Saving FastRP FAISS index...")
        try:
            faiss.write_index(self.faiss_index, self.FAISS_INDEX_PATH)
            with open(self.HOTEL_MAPPING_PATH, 'wb') as f:
                pickle.dump({
                    'hotel_id_to_idx': self.hotel_id_to_idx,
                    'idx_to_hotel_id': self.idx_to_hotel_id
                }, f)
            print(f"  Saved to {self.FAISS_INDEX_PATH}")
            return True
        except Exception as e:
            print(f"  Error saving: {e}")
            return False

    def load_faiss_index(self):
        """Load FAISS index and mappings from disk"""
        print("\n Loading FastRP FAISS index...")
        try:
            if not os.path.exists(self.FAISS_INDEX_PATH):
                print(f"  Index file not found: {self.FAISS_INDEX_PATH}")
                return False

            self.faiss_index = faiss.read_index(self.FAISS_INDEX_PATH)
            with open(self.HOTEL_MAPPING_PATH, 'rb') as f:
                mappings = pickle.load(f)
                self.hotel_id_to_idx = mappings['hotel_id_to_idx']
                self.idx_to_hotel_id = mappings['idx_to_hotel_id']

            print(f"  Index loaded ({self.faiss_index.ntotal} vectors)")
            print(f"  Mappings loaded ({len(self.hotel_id_to_idx)} hotels)")
            return True
        except Exception as e:
            print(f"  Error loading: {e}")
            return False

    def find_similar_hotels(self, hotel_name, top_k=5):
        """Find similar hotels using FastRP embeddings"""
        if self.faiss_index is None:
            print("  No FAISS index loaded!")
            return []

        # Get hotel ID
        query = """
        MATCH (h:Hotel)
        WHERE toLower(h.name) CONTAINS toLower($name)
        RETURN h.hotel_id as hotel_id, h.name as name
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query, name=hotel_name)
            record = result.single()

        if not record:
            print(f"  Hotel '{hotel_name}' not found")
            return []

        hotel_id = record['hotel_id']
        if hotel_id not in self.hotel_id_to_idx:
            print(f"  Hotel '{hotel_name}' has no FastRP embedding")
            return []

        # Search FAISS
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
        details_query = """
        UNWIND $hotel_ids as hotel_id
        MATCH (h:Hotel {hotel_id: hotel_id})-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(country:Country)
        RETURN h.hotel_id as hotel_id, h.name as name,
               c.name as city, country.name as country,
               h.star_rating as star_rating
        """
        with self.driver.session() as session:
            result = session.run(details_query, hotel_ids=similar_hotels)
            hotel_details = {r['hotel_id']: r for r in result.data()}

        final_results = []
        for hid, score in zip(similar_hotels, scores):
            if hid in hotel_details:
                details = hotel_details[hid]
                final_results.append({
                    'hotel': details['name'],
                    'hotel_id': hid,
                    'city': details['city'],
                    'country': details['country'],
                    'score': float(score),
                    'star_rating': details.get('star_rating')
                })
        return final_results

    def setup_embeddings(self):
        """Complete FastRP setup pipeline"""
        print("\n" + "="*60)
        print(" FASTRP EMBEDDING SETUP")
        print("="*60)

        try:
            # Step 1: Create graph projection
            print("\n Step 1: Creating graph projection...")
            self.create_graph()

            # Step 2: Run FastRP
            print("\n Step 2: Running FastRP algorithm...")
            self.run_fastrp()

            # Step 3: Create vector index
            print("\n Step 3: Creating vector index...")
            self.create_vector_index()

            # Step 4: Build FAISS index
            print("\n Step 4: Building FAISS index...")
            if not self.build_faiss_index():
                print("\n Setup failed: Could not build FAISS index")
                return False

            # Step 5: Save FAISS index
            if not self.save_faiss_index():
                print("\n Warning: Could not save FAISS index")

            print("\n" + "="*60)
            print(" FASTRP EMBEDDING SETUP COMPLETE")
            print("="*60)
            return True

        except Exception as e:
            print(f"\nSetup failed: {e}")
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

    VALID_MODELS = ['node2vec', 'fastrp']

    def __init__(self, driver, model_type='node2vec', auto_initialize=True):
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
        else:  # 'fastrp'
            self._retriever = FastRPEmbeddingRetriever(self.driver)
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
            return True

        # No existing index - check if embeddings exist in Neo4j
        if self._retriever.verify_embeddings_exist():
            # Embeddings exist, just build FAISS index
            print(f"  Embeddings exist, building FAISS index...")
            if self._retriever.build_faiss_index():
                self._retriever.save_faiss_index()
                self._initialized = True
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
                        print(f"   Avg: {r['avg_score']:.1f} | Clean: {r.get('cleanliness', 0):.1f} | Comfort: {r.get('comfort', 0):.1f} | Facilities: {r.get('facilities', 0):.1f} | Location: {r.get('location', 0):.1f}")
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
