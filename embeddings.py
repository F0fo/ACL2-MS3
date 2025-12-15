

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
    
    # FAISS files (per node type)
    FAISS_DIR = "faiss_indices"
    
    def __init__(self, driver):
        """Initialize with Neo4j driver"""
        self.driver = driver
        
        # FAISS indices per node type
        self.faiss_node2vec = {}  # {label: faiss_index}
        self.faiss_text = {}      # {label: faiss_index}
        
        # Mappings per node type
        self.node_id_to_idx = {}  # {label: {node_id: idx}}
        self.idx_to_node_id = {}  # {label: {idx: node_id}}
        
        # Text model for creating embeddings
        print(" Loading Sentence Transformer...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" ✓ Model loaded")

        #Create FAISS directory 
        os.makedirs(self.FAISS_DIR, exist_ok=True)
    
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
    
    def build_faiss_indices(self, label):
        """Build TWO FAISS indices: Node2Vec and Text"""
        print(f"\n Building FAISS indices for {label}...")

        primary_key = self.get_primary_key(label)
        
        # First check what embeddings exist
        check_query = f"""
        MATCH (n:{label})
        RETURN 
            count(n) as total,
            sum(CASE WHEN n.{self.NODE2VEC_PROP} IS NOT NULL THEN 1 ELSE 0 END) as with_node2vec,
            sum(CASE WHEN n.{self.TEXT_EMBEDDING_PROP} IS NOT NULL THEN 1 ELSE 0 END) as with_text
        """
        with self.driver.session() as session:
            result = session.run(check_query)
            stats = result.single()

        print(f"     {label} embedding status:")
        print(f"     Total nodes: {stats['total']}")
        print(f"     With Node2Vec: {stats['with_node2vec']}")
        print(f"     With Text: {stats['with_text']}")
        
        if stats['with_node2vec'] == 0:
            print(f"   No Node2Vec embeddings found for {label}!")
            print(f"   Run Node2Vec algorithm first:")
            print(f"   python embedding_retriever.py --setup")
            return False
        
        if stats['with_text'] == 0:
            print(f"   No text embeddings found for {label}!")
            return False

        # Fetch both embeddings from Neo4j
        query = f"""
        MATCH (n:{label})
        WHERE n.{self.NODE2VEC_PROP} IS NOT NULL 
            AND n.{self.TEXT_EMBEDDING_PROP} IS NOT NULL
        RETURN n.{primary_key} as node_id,
            n.{self.NODE2VEC_PROP} as node2vec_embedding,
            n.{self.TEXT_EMBEDDING_PROP} as text_embedding
        ORDER BY node_id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()
        
        if not records:
            print(f"  No {label} with both embeddings found")
            return False
        
        print(f"  Found {len(records)} hotels with both embeddings")
        
        # Prepare embedding matrices
        node2vec_list = []
        text_list = []
        
        for idx, record in enumerate(records):
            node_id = record['node_id']
            
            node2vec_emb = np.array(record['node2vec_embedding'], dtype='float32')
            text_emb = np.array(record['text_embedding'], dtype='float32')
            
            node2vec_list.append(node2vec_emb)
            text_list.append(text_emb)
            
            # Create mappings
            self.node_id_to_idx[label][node_id] = idx
            self.idx_to_node_id[label][idx] = node_id
        
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
        
        # Index 1: Node2Vec 
        self.faiss_node2vec[label] = faiss.IndexFlatIP(self.NODE2VEC_DIM)
        self.faiss_node2vec[label].add(node2vec_matrix)
        print(f"    ✓ Node2Vec index: {self.faiss_node2vec[label].ntotal} vectors")
        
        # Index 2: Text
        self.faiss_text[label] = faiss.IndexFlatIP(self.TEXT_DIM)
        self.faiss_text[label].add(text_matrix)
        print(f"    ✓ Text index: {self.faiss_text[label].ntotal} vectors")
        
        return True
    
    def save_faiss_indices(self, label):
        """Save both FAISS indices to disk"""
        print("\n Saving FAISS indices for {label}...")
        
        try:
            # File paths
            node2vec_path = os.path.join(self.FAISS_DIR, f"{label}_node2vec.index")
            text_path = os.path.join(self.FAISS_DIR, f"{label}_text.index")
            mapping_path = os.path.join(self.FAISS_DIR, f"{label}_mapping.pkl")
            
            # Save indices
            faiss.write_index(self.faiss_node2vec[label], node2vec_path)
            print(f"  ✓ Node2Vec saved to {node2vec_path}")
            
            faiss.write_index(self.faiss_text[label], text_path)
            print(f"  ✓ Text saved to {text_path}")
            
            # Save mappings
            with open(mapping_path, 'wb') as f:
                pickle.dump({
                    'node_id_to_idx': self.node_id_to_idx[label],
                    'idx_to_node_id': self.idx_to_node_id[label]
                }, f)
            print(f"  ✓ Mappings saved to {mapping_path}")
            
            return True
        except Exception as e:
            print(f"   Error saving: {e}")
            return False
    
    def load_faiss_indices(self, label):
        """Load both FAISS indices from disk"""
        print("\n Loading FAISS indices...")
        
         # File paths
        node2vec_path = os.path.join(self.FAISS_DIR, f"{label}_node2vec.index")
        text_path = os.path.join(self.FAISS_DIR, f"{label}_text.index")
        mapping_path = os.path.join(self.FAISS_DIR, f"{label}_mapping.pkl")
        
        required_files = [node2vec_path, text_path, mapping_path]
        
        for file in required_files:
            if not os.path.exists(file):
                print(f"  Missing: {file}")
                return False
        
        try:
            # Load indices
            self.faiss_node2vec[label] = faiss.read_index(node2vec_path)
            print(f"  ✓ Node2Vec loaded ({self.faiss_node2vec[label].ntotal} vectors)")
            
            self.faiss_text[label] = faiss.read_index(text_path)
            print(f"  ✓ Text loaded ({self.faiss_text[label].ntotal} vectors)")
            
            # Load mappings
            with open(mapping_path, 'rb') as f:
                mappings = pickle.load(f)
            self.node_id_to_idx[label] = mappings['node_id_to_idx']
            self.idx_to_node_id[label] = mappings['idx_to_node_id']
            print(f"  ✓ Mappings loaded ({len(self.node_id_to_idx[label])} nodes)")
            
            return True
        except Exception as e:
            print(f" Error loading: {e}")
            return False
    
    def get_node_labels(self):
        """Get all node labels in the database"""
        query = "CALL db.labels()"
        with self.driver.session() as session:
            result = session.run(query)
            labels = [record['label'] for record in result]
        return labels
    
    def get_node_properties(self, label):
        """Get properties for a node label"""
        query = f"""
        MATCH (n:{label})
        WITH n LIMIT 1
        RETURN keys(n) as properties
        """
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record['properties'] if record else []
    
    def get_text_properties(self, label):
        """
        Determine which properties to use for text embeddings
        Automatically finds text properties based on node type
        """
        props = self.get_node_properties(label)
        
        # Common text property patterns
        text_props = []
        for prop in props:
            # Skip embedding properties
            if 'embedding' in prop.lower():
                continue
            # Include texyt-like properties
            if any(keyword in prop.lower() for keyword in 
                   ['name', 'text', 'description', 'title', 'type', 'gender', 'age']):
                text_props.append(prop)
        
        return text_props
    
    def get_primary_key(self, label):
        """Determine primary key for a node label"""
        props = self.get_node_properties(label)
        
        # Common primary key patterns
        if f'{label.lower()}_id' in props:
            return f'{label.lower()}_id'
        elif 'id' in props:
            return 'id'
        elif 'name' in props:
            return 'name'
        else:
            return props[0] if props else None

    def generate_text_embeddings(self, label, batch_size=32):
        """
        Generate text embeddings for all nodes of a given label
        """
        print(f"\n Generating text embeddings for label: '{label}'")

        text_props = self.get_text_properties(label)
        primary_key = self.get_primary_key(label)

        if not text_props or not primary_key:
            print(f" Cannot determine text properties or primary key for label '{label}'")
            return
        print(f" Using text properties: {text_props}")
        print(f" Using primary key: {primary_key}")

        # Fetch nodes
        props_str = ', '.join([f'n.{prop}' for prop in text_props])
        query = f"""
        MATCH (n:{label})
        RETURN n.{primary_key} as node_id, {props_str}
        ORDER BY node_id
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()
        if not records:
            print(f" No nodes found for label '{label}'")
            return False
        print(f" Processing {len(records)} nodes...")

        embeddings_to_save = []

        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]

            # Create text representations
            node_texts = []
            node_ids = []

            for record in batch:
                node_id = record['node_id']
                text_parts = []

                for prop in text_props:
                    value = record.get(prop)
                    if value:
                        text_parts.append(f"{prop}: {str(value)[:200]}")  # Limit length

                combined_text = ". ".join(text_parts)

                node_texts.append(combined_text)
                node_ids.append(node_id)

            # Generate embeddings for batch
            batch_embeddings = self.text_model.encode(
                node_texts,
                show_progress_bar=False,
                batch_size=batch_size
            )

            # Prepare for Neo4j
            for node_id, embedding in zip(node_ids, batch_embeddings):
                embeddings_to_save.append({
                    'node_id': node_id,
                    'embedding': embedding.tolist()
                })

            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {i + len(batch)}/{len(records)}...")
        # Save to Neo4j
        print(f"  Saving text embeddings to Neo4j for label '{label}'...")

        with self.driver.session() as session:
            session.run(f"""
                UNWIND $embeddings as emb
                MATCH (n:{label} {{{primary_key}: emb.node_id}})
                SET n.{self.TEXT_EMBEDDING_PROP} = emb.embedding
            """, embeddings=embeddings_to_save)

        print(f"  ✓ Text embeddings created for {len(embeddings_to_save)} nodes of label '{label}'")
        return True

        
    
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
    
    def find_similar_nodes(self, label, node_id, top_k=5):
        """
        Find similar nodes using Node2Vec embeddings
        Works for any node type: Hotel, City, Country, Traveller, Review
        """
        print(f"\n Finding similar {label} nodes to ID: {node_id}")
        
        if label not in self.faiss_node2vec:
            print(f"    No index loaded for {label}")
            return []
        
        # Get index position
        if node_id not in self.node_id_to_idx[label]:
            print(f"    Node ID not in index")
            return []
        
        idx = self.node_id_to_idx[label][node_id]
        
        # Get Node2Vec embedding
        query_vector = self.faiss_node2vec[label].reconstruct(idx).reshape(1, -1)
        
        # Search
        distances, indices = self.faiss_node2vec[label].search(query_vector, top_k + 1)
        
        # Get results (skip query node)
        similar_node_ids = []
        scores = []
        
        for i, score in zip(indices[0], distances[0]):
            result_node_id = self.idx_to_node_id[label][i]
            if result_node_id == node_id:
                continue
            similar_node_ids.append(result_node_id)
            scores.append(float(score))
            if len(similar_node_ids) >= top_k:
                break
        
        return self._enrich_results(label, similar_node_ids, scores, "Node2Vec similarity")
    
    def search_by_query(self,label, query_text, top_k=5):
        
        print(f"\n Searching by query: '{query_text}'")
        
        if label not in self.faiss_text:
            print(f"    No text index loaded for {label}")
            return []

        # Convert query to text embedding (SAME model as hotels!)
        query_embedding = self.text_model.encode([query_text])[0]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Normalize
        faiss.normalize_L2(query_embedding)
        
        # Search TEXT index (both in same 384-dim space!)
        distances, indices = self.faiss_text.search(query_embedding, top_k)
        
        # Get node IDs
        node_ids = [self.idx_to_node_id[label][idx] for idx in indices[0]]
        scores = distances[0].tolist()
        
        # Enrich with details
        return self._enrich_results(label, node_ids, scores, "Semantic text match")
    
    def _enrich_results(self, label, node_ids, scores, method):
        """Add node details to results"""
        if not node_ids:
            return []
        
        primary_key = self.get_primary_key(label)
        text_props = self.get_text_properties(label)

        return_props = [f'n.{primary_key} as node_id']
        for prop in text_props:
            return_props.append(f'n.{prop} as {prop}')
        return_props_str = ', '.join(return_props)

        query = f"""
        UNWIND $node_ids as node_id
        MATCH (n:{label} {{{primary_key}: node_id}})
        RETURN {return_props_str}
        """

        with self.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            node_details = {r['node_id']: r for r in result.data()}
        
        # Combine
        results = []
        for node_id, score in zip(node_ids, scores):
            if node_id in node_details:
                node = dict(node_details[node_id])
                node['score'] = score
                node['method'] = method
                results.append(node)
        
        return results
    
    def setup_all(self, labels=None):
        """Complete setup: Generate text embeddings + build indices"""
        print("\n" + "="*70)
        print("DUAL EMBEDDING SETUP")
        print("="*70)
        
        # Get labels
        if labels is None:
            labels = self.get_node_labels()
        
        print(f"\n Processing labels: {labels}")
        
        success_count = 0
        
        for label in labels:
            print(f"\n{'='*70}")
            print(f"Processing {label}")
            print(f"{'='*70}")
            
            try:
                # Step 1: Generate text embeddings (if not exists)
                print("\n Checking for text embeddings...")
                query = f"MATCH (n:{label}) WHERE n.{self.TEXT_EMBEDDING_PROP} IS NOT NULL RETURN count(n) as count"
                with self.driver.session() as session:
                    result = session.run(query)
                    count = result.single()['count']
                
                if count == 0:
                    print(f"  No text embeddings found. Generating...")
                    if not self.generate_text_embeddings(label):
                        print(f"  Skipping {label} - no text properties")
                        continue
                else:
                    print(f"  ✓ Found {count} {label} nodes with text embeddings")
                
                # Step 2: Build FAISS indices
                if not self.build_faiss_indices(label):
                    continue
                
                # Step 3: Save indices
                if not self.save_faiss_indices(label):
                    continue
                
                success_count += 1
                
            except Exception as e:
                print(f"  Error processing {label}: {e}")
                continue
        
        print("\n" + "="*70)
        print(f" SETUP COMPLETE: {success_count}/{len(labels)} labels processed")
        print("="*70)
        
        return success_count > 0

def main():
    """Test generalized dual embedding retriever"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generalized Dual Embedding Retriever')
    parser.add_argument('--setup', action='store_true', help='Setup embeddings for all nodes')
    parser.add_argument('--label', type=str, help='Node label (Hotel, City, etc.)')
    parser.add_argument('--node-id', help='Node ID for similarity search')
    parser.add_argument('--query', type=str, help='Text query for semantic search')
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
        
        # Search
        if args.label:
            # Load index for this label
            if not retriever.load_faiss_indices(args.label):
                print(f"\n Run --setup first!")
                db.close()
                exit(1)
            
            # Node similarity
            if args.node_id:
                results = retriever.find_similar_nodes(
                    args.label,
                    args.node_id,
                    top_k=args.top_k
                )
                print_results(results, f"Similar {args.label} nodes")
            
            # Query search
            elif args.query:
                results = retriever.search_by_query(
                    args.label,
                    args.query,
                    top_k=args.top_k
                )
                print_results(results, f"{args.label} query results")
        
        # Demo
        if not any([args.setup, args.label]):
            parser.print_help()
            print("\n" + "="*70)
            print("EXAMPLES:")
            print("="*70)
            print("\n# Setup all node types")
            print("python generalized_dual_retriever.py --setup")
            print("\n# Find similar hotels (Node2Vec)")
            print("python generalized_dual_retriever.py --label Hotel --node-id 123")
            print("\n# Search hotels by query (Text)")
            print("python generalized_dual_retriever.py --label Hotel --query 'luxury spa'")
            print("\n# Search cities by query")
            print("python generalized_dual_retriever.py --label City --query 'beach paradise'")
            print("="*70)
        
        db.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def print_results(results, title):
    """Pretty print results"""
    print("\n" + "="*70)
    print(f"📋 {title}")
    print("="*70)
    
    if not results:
        print("  No results found")
        return
    
    for i, node in enumerate(results, 1):
        print(f"\n{i}. Node ID: {node.get('node_id')}")
        print(f"   Label: {node.get('label')}")
        print(f"   Score: {node['score']:.3f}")
        print(f"   Method: {node['method']}")
        
        # Print all properties except system ones
        for key, value in node.items():
            if key not in ['node_id', 'label', 'score', 'method']:
                print(f"   {key}: {value}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
