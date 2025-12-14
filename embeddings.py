

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from db_manager import DBManager
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer


class DualEmbeddingRetriever:

    
    # Configuration
    NODE2VEC_PROP = "node2vecEmbedding"
    TEXT_EMBEDDING_PROP = "textEmbedding"
    NODE2VEC_DIM = 128
    TEXT_DIM = 384  # all-MiniLM-L6-v2
    
    # FAISS files
    FAISS_NODE2VEC_PATH = "hotel_node2vec.index"
    FAISS_TEXT_PATH = "hotel_text.index"
    HOTEL_MAPPING_PATH = "hotel_mapping.pkl"
    
    def __init__(self, driver):
        """Initialize with Neo4j driver"""
        self.driver = driver
        
        # Two FAISS indices
        self.faiss_node2vec = None  # For hotel similarity
        self.faiss_text = None       # For query matching
        
        # Mappings
        self.hotel_id_to_idx = {}
        self.idx_to_hotel_id = {}
        
        # Text model for creating embeddings
        print(" Loading Sentence Transformer...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" ✓ Model loaded")
    
    def generate_text_embeddings(self, batch_size=32):
        """
        Generate text embeddings for all hotels
        Combines hotel name + reviews into text representation
        """
        print("\n Generating text embeddings for hotels...")
        
        # Fetch hotels and their reviews
        query = """
        MATCH (h:Hotel)
        OPTIONAL MATCH (r:Review)-[:REVIEWED]->(h)
        WITH h, 
             collect(r.text)[0..10] as reviews,  // First 10 reviews
             h.name as name,
             h.star_rating as stars
        RETURN h.hotel_id as hotel_id,
               name,
               stars,
               reviews
        ORDER BY hotel_id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()
        
        print(f" Processing {len(records)} hotels...")
        
        embeddings_to_save = []
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            
            # Create text representations
            hotel_texts = []
            hotel_ids = []
            
            for record in batch:
                hotel_id = record['hotel_id']
                name = record['name']
                stars = record['stars'] if record['stars'] else 0
                reviews = record['reviews']
                
                # Combine into a single text representation
                # This captures what the hotel is about
                text_parts = [
                    f"Hotel name: {name}",
                    f"Star rating: {stars} stars"
                ]
                
                if reviews:
                    # Add review content
                    review_text = " ".join([r for r in reviews if r])
                    text_parts.append(f"Reviews: {review_text[:500]}")  # Limit length
                
                combined_text = ". ".join(text_parts)
                
                hotel_texts.append(combined_text)
                hotel_ids.append(hotel_id)
            
            # Generate embeddings for batch
            batch_embeddings = self.text_model.encode(
                hotel_texts, 
                show_progress_bar=False,
                batch_size=batch_size
            )
            
            # Prepare for Neo4j
            for hotel_id, embedding in zip(hotel_ids, batch_embeddings):
                embeddings_to_save.append({
                    'hotel_id': hotel_id,
                    'embedding': embedding.tolist()
                })
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {i + len(batch)}/{len(records)}...")
        
        # Save to Neo4j
        print("  Saving text embeddings to Neo4j...")
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $embeddings as emb
                MATCH (h:Hotel {hotel_id: emb.hotel_id})
                SET h.textEmbedding = emb.embedding
            """, embeddings=embeddings_to_save)
        
        print(f"  ✓ Text embeddings created for {len(embeddings_to_save)} hotels")
    
    def build_faiss_indices(self):
        """Build TWO FAISS indices: Node2Vec and Text"""
        print("\n Building FAISS indices...")
        
        # Fetch both embeddings from Neo4j
        query = f"""
        MATCH (h:Hotel)
        WHERE h.{self.NODE2VEC_PROP} IS NOT NULL 
          AND h.{self.TEXT_EMBEDDING_PROP} IS NOT NULL
        RETURN h.hotel_id as hotel_id,
               h.{self.NODE2VEC_PROP} as node2vec_embedding,
               h.{self.TEXT_EMBEDDING_PROP} as text_embedding
        ORDER BY hotel_id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()
        
        if not records:
            print("  No hotels with both embeddings found")
            return False
        
        print(f"  Found {len(records)} hotels with both embeddings")
        
        # Prepare embedding matrices
        node2vec_list = []
        text_list = []
        
        for idx, record in enumerate(records):
            hotel_id = record['hotel_id']
            
            node2vec_emb = np.array(record['node2vec_embedding'], dtype='float32')
            text_emb = np.array(record['text_embedding'], dtype='float32')
            
            node2vec_list.append(node2vec_emb)
            text_list.append(text_emb)
            
            # Create mappings
            self.hotel_id_to_idx[hotel_id] = idx
            self.idx_to_hotel_id[idx] = hotel_id
        
        # Stack into matrices
        node2vec_matrix = np.vstack(node2vec_list)
        text_matrix = np.vstack(text_list)
        
        print(f" Node2Vec matrix: {node2vec_matrix.shape}")
        print(f" Text matrix: {text_matrix.shape}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(node2vec_matrix)
        faiss.normalize_L2(text_matrix)
        
        # Build FAISS indices
        print(" Building indices...")
        
        # Index 1: Node2Vec (for hotel similarity)
        self.faiss_node2vec = faiss.IndexFlatIP(self.NODE2VEC_DIM)
        self.faiss_node2vec.add(node2vec_matrix)
        print(f"    ✓ Node2Vec index: {self.faiss_node2vec.ntotal} vectors")
        
        # Index 2: Text (for query matching)
        self.faiss_text = faiss.IndexFlatIP(self.TEXT_DIM)
        self.faiss_text.add(text_matrix)
        print(f"    ✓ Text index: {self.faiss_text.ntotal} vectors")
        
        return True
    
    def save_faiss_indices(self):
        """Save both FAISS indices to disk"""
        print("\n Saving FAISS indices...")
        
        try:
            # Save Node2Vec index
            faiss.write_index(self.faiss_node2vec, self.FAISS_NODE2VEC_PATH)
            print(f"  ✓ Node2Vec saved to {self.FAISS_NODE2VEC_PATH}")
            
            # Save Text index
            faiss.write_index(self.faiss_text, self.FAISS_TEXT_PATH)
            print(f"  ✓ Text saved to {self.FAISS_TEXT_PATH}")
            
            # Save mappings
            with open(self.HOTEL_MAPPING_PATH, 'wb') as f:
                pickle.dump({
                    'hotel_id_to_idx': self.hotel_id_to_idx,
                    'idx_to_hotel_id': self.idx_to_hotel_id
                }, f)
            print(f"  ✓ Mappings saved to {self.HOTEL_MAPPING_PATH}")
            
            return True
        except Exception as e:
            print(f"  ❌ Error saving: {e}")
            return False
    
    def load_faiss_indices(self):
        """Load both FAISS indices from disk"""
        print("\n Loading FAISS indices...")
        
        required_files = [
            self.FAISS_NODE2VEC_PATH,
            self.FAISS_TEXT_PATH,
            self.HOTEL_MAPPING_PATH
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"  Missing: {file}")
                return False
        
        try:
            # Load indices
            self.faiss_node2vec = faiss.read_index(self.FAISS_NODE2VEC_PATH)
            print(f"  ✓ Node2Vec loaded ({self.faiss_node2vec.ntotal} vectors)")
            
            self.faiss_text = faiss.read_index(self.FAISS_TEXT_PATH)
            print(f"  ✓ Text loaded ({self.faiss_text.ntotal} vectors)")
            
            # Load mappings
            with open(self.HOTEL_MAPPING_PATH, 'rb') as f:
                mappings = pickle.load(f)
            self.hotel_id_to_idx = mappings['hotel_id_to_idx']
            self.idx_to_hotel_id = mappings['idx_to_hotel_id']
            print(f"  ✓ Mappings loaded ({len(self.hotel_id_to_idx)} hotels)")
            
            return True
        except Exception as e:
            print(f" Error loading: {e}")
            return False
    
    # ==================== SEARCH METHODS ====================
    
    def find_similar_hotels(self, hotel_name, top_k=5):
        
        print(f"\n Searching similar hotels to: '{hotel_name}'")
        
        # Get hotel_id
        query = "MATCH (h:Hotel {name: $name}) RETURN h.hotel_id as hotel_id"
        
        with self.driver.session() as session:
            result = session.run(query, name=hotel_name)
            record = result.single()
            
            if not record:
                print(f"  Hotel not found")
                return []
            
            hotel_id = record['hotel_id']
        
        # Get index position
        if hotel_id not in self.hotel_id_to_idx:
            print(f"  Hotel not in index")
            return []
        
        idx = self.hotel_id_to_idx[hotel_id]
        
        # Get Node2Vec embedding
        query_vector = self.faiss_node2vec.reconstruct(idx).reshape(1, -1)
        
        # Search Node2Vec index
        distances, indices = self.faiss_node2vec.search(query_vector, top_k + 1)
        
        # Get results (skip query hotel)
        similar_hotel_ids = []
        scores = []
        
        for i, score in zip(indices[0], distances[0]):
            result_hotel_id = self.idx_to_hotel_id[i]
            if result_hotel_id == hotel_id:
                continue
            similar_hotel_ids.append(result_hotel_id)
            scores.append(float(score))
            if len(similar_hotel_ids) >= top_k:
                break
        
        # Enrich with details
        return self._enrich_results(similar_hotel_ids, scores, "Node2Vec similarity")
    
    def search_by_query(self, query_text, top_k=5):
        
        print(f"\n Searching by query: '{query_text}'")
        
        # Convert query to text embedding (SAME model as hotels!)
        query_embedding = self.text_model.encode([query_text])[0]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Normalize
        faiss.normalize_L2(query_embedding)
        
        # Search TEXT index (both in same 384-dim space!)
        distances, indices = self.faiss_text.search(query_embedding, top_k)
        
        # Get hotel IDs
        hotel_ids = [self.idx_to_hotel_id[idx] for idx in indices[0]]
        scores = distances[0].tolist()
        
        # Enrich with details
        return self._enrich_results(hotel_ids, scores, "Semantic text match")
    
    def _enrich_results(self, hotel_ids, scores, method):
        """Add hotel details to results"""
        if not hotel_ids:
            return []
        
        query = """
        UNWIND $hotel_ids as hotel_id
        MATCH (h:Hotel {hotel_id: hotel_id})
        OPTIONAL MATCH (r:Review)-[:REVIEWED]->(h)
        WITH h, avg(r.score_overall) as avg_rating, count(r) as review_count
        RETURN h.hotel_id as hotel_id,
               h.name as name,
               h.star_rating as star_rating,
               avg_rating,
               review_count
        """
        
        with self.driver.session() as session:
            result = session.run(query, hotel_ids=hotel_ids)
            hotel_details = {r['hotel_id']: r for r in result.data()}
        
        # Combine
        results = []
        for hotel_id, score in zip(hotel_ids, scores):
            if hotel_id in hotel_details:
                hotel = dict(hotel_details[hotel_id])
                hotel['score'] = score
                hotel['method'] = method
                results.append(hotel)
        
        return results
    
    def setup_all(self):
        """Complete setup: Generate text embeddings + build indices"""
        print("\n" + "="*70)
        print("DUAL EMBEDDING SETUP")
        print("="*70)
        
        try:
            # Step 1: Generate text embeddings (if not exists)
            print("\n Checking for text embeddings...")
            query = "MATCH (h:Hotel) WHERE h.textEmbedding IS NOT NULL RETURN count(h) as count"
            with self.driver.session() as session:
                result = session.run(query)
                count = result.single()['count']
            
            if count == 0:
                print("  No text embeddings found. Generating...")
                self.generate_text_embeddings()
            else:
                print(f"  ✓ Found {count} hotels with text embeddings")
            
            # Step 2: Build FAISS indices
            if not self.build_faiss_indices():
                return False
            
            # Step 3: Save indices
            if not self.save_faiss_indices():
                return False
            
            print("\n" + "="*70)
            print(" DUAL EMBEDDING SETUP COMPLETE")
            print("="*70)
            print("\n You now have:")
            print(f"  • Node2Vec index ({self.NODE2VEC_DIM}D) - for hotel similarity")
            print(f"  • Text index ({self.TEXT_DIM}D) - for query matching")
            print("\n Usage:")
            print('  retriever.find_similar_hotels("Grand Palace", top_k=5)')
            print('  retriever.search_by_query("luxury hotels with spa", top_k=5)')
            
            return True
            
        except Exception as e:
            print(f"\n Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Test dual embedding retriever"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dual Embedding Hotel Retriever')
    parser.add_argument('--setup', action='store_true', help='Setup embeddings and indices')
    parser.add_argument('--hotel', type=str, help='Find similar hotels')
    parser.add_argument('--query', type=str, help='Search by text query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    try:
        # Initialize
        print("🔌 Connecting to Neo4j...")
        db = DBManager()
        
        retriever = DualEmbeddingRetriever(db.driver)
        
        # Setup
        if args.setup:
            success = retriever.setup_all()
            if not success:
                db.close()
                exit(1)
        
        # Load indices
        if args.hotel or args.query:
            if not retriever.load_faiss_indices():
                print("\n  Run --setup first!")
                db.close()
                exit(1)
        
        # Search by hotel similarity
        if args.hotel:
            results = retriever.find_similar_hotels(args.hotel, top_k=args.top_k)
            print_results(results, f"Similar to '{args.hotel}'")
        
        # Search by text query
        if args.query:
            results = retriever.search_by_query(args.query, top_k=args.top_k)
            print_results(results, f"Query: '{args.query}'")
        
        # Demo if no args
        if not any([args.setup, args.hotel, args.query]):
            parser.print_help()
            print("\n" + "="*70)
            print("EXAMPLES:")
            print("="*70)
            print("\n# Setup (run once)")
            print("python dual_embedding_retriever.py --setup")
            print("\n# Hotel similarity (Node2Vec)")
            print("python dual_embedding_retriever.py --hotel 'Grand Palace'")
            print("\n# Query search (Text embeddings)")
            print("python dual_embedding_retriever.py --query 'luxury hotels with spa'")
            print("="*70)
        
        db.close()
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def print_results(results, title):
    """Pretty print results"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)
    
    if not results:
        print("  No results found")
        return
    
    for i, hotel in enumerate(results, 1):
        print(f"\n{i}. {hotel['name']}")
        print(f"    Star Rating: {hotel.get('star_rating', 'N/A')}")
        if hotel.get('avg_rating'):
            print(f"    Avg Review: {hotel['avg_rating']:.2f}")
        print(f"    Score: {hotel['score']:.3f}")
        print(f"    Method: {hotel['method']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()