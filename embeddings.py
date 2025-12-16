

import pandas as pd
import numpy as np
import re
import statistics
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
        self.faiss_combined = {}  # {label: faiss_index}  -- text + numeric
        
        # Mappings per node type
        self.node_id_to_idx = {}  # {label: {node_id: idx}}
        self.idx_to_node_id = {}  # {label: {idx: node_id}}
        
        # Numeric metadata per node type
        # self.numeric_props[label] = [prop1, prop2, ...]
        # self.numeric_stats[label] = {'min': [...],'max': [...],'mean': [...]}
        self.numeric_props = {}
        self.numeric_stats = {}

        # Text->Node2Vec mapping weights per label
        # self.text2node_W[label] = matrix of shape (text_dim, node2vec_dim)
        self.text2node_W = {}
        
        # Text model for creating embeddings
        print(" Loading Sentence Transformer...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded")

        #Create FAISS directory
        os.makedirs(self.FAISS_DIR, exist_ok=True)

    def generate_hotel_text_embeddings(self, batch_size=32):
        """
        Generate text embeddings specifically for hotels.
        Combines hotel name + star rating + reviews into text representation.
        This method provides richer embeddings for hotels than the generic method.
        """
        print("\n Generating text embeddings for hotels (with reviews)...")

        # Fetch hotels and their reviews
        query = """
        MATCH (h:Hotel)
        OPTIONAL MATCH (r:Review)-[:REVIEWED]->(h)
        WITH h,
             collect(r.text)[0..10] as reviews,
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
                text_parts = [
                    f"Hotel name: {name}",
                    f"Star rating: {stars} stars"
                ]

                if reviews:
                    # Add review content (limit to 500 chars)
                    review_text = " ".join([r for r in reviews if r])
                    text_parts.append(f"Reviews: {review_text[:500]}")

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

        print(f"  Text embeddings created for {len(embeddings_to_save)} hotels")
        return True

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

        # Fetch numeric properties for this label
        numeric_props = self.get_numeric_properties(label)
        self.numeric_props[label] = numeric_props

        # Build dynamic return string if numeric props exist
        numeric_return = ''
        if numeric_props:
            numeric_return = ', ' + ', '.join([f'n.{p} as {p}' for p in numeric_props])

        # Fetch both embeddings + numeric props from Neo4j
        query = f"""
        MATCH (n:{label})
        WHERE n.{self.NODE2VEC_PROP} IS NOT NULL 
            AND n.{self.TEXT_EMBEDDING_PROP} IS NOT NULL
        RETURN n.{primary_key} as node_id,
            n.{self.NODE2VEC_PROP} as node2vec_embedding,
            n.{self.TEXT_EMBEDDING_PROP} as text_embedding{numeric_return}
        ORDER BY node_id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()
        
        if not records:
            print(f"  No {label} with both embeddings found")
            return False
        
        print(f"  Found {len(records)} {label} with both embeddings")
        
        # Prepare embedding matrices
        node2vec_list = []
        text_list = []
        numeric_list = []
        
        # Initialize mappings
        self.node_id_to_idx[label] = {}  
        self.idx_to_node_id[label] = {}  

        for idx, record in enumerate(records):
            node_id = record['node_id']
            
            node2vec_emb = np.array(record['node2vec_embedding'], dtype='float32')
            text_emb = np.array(record['text_embedding'], dtype='float32')
            
            node2vec_list.append(node2vec_emb)
            text_list.append(text_emb)

            # Numeric props
            if numeric_props:
                num_vals = []
                for p in numeric_props:
                    v = record.get(p)
                    if v is None:
                        num_vals.append(np.nan)
                    else:
                        try:
                            num_vals.append(float(v))
                        except Exception:
                            num_vals.append(np.nan)
                numeric_list.append(num_vals)
            
            # Create mappings
            self.node_id_to_idx[label][node_id] = idx
            self.idx_to_node_id[label][idx] = node_id
        
        # Stack into matrices
        node2vec_matrix = np.vstack(node2vec_list)
        text_matrix = np.vstack(text_list)
        
        print(f" Node2Vec matrix: {node2vec_matrix.shape}")
        print(f" Text matrix: {text_matrix.shape}")
        
        # Normalize text/node2vec for cosine similarity
        faiss.normalize_L2(node2vec_matrix)
        faiss.normalize_L2(text_matrix)

        # If numeric props exist, create normalized numeric matrix and a combined index (text + numeric)
        combined_matrix = None
        if numeric_props:
            numeric_matrix = np.array(numeric_list, dtype='float32')
            # Handle NaNs by replacing with column means
            col_means = np.nanmean(numeric_matrix, axis=0)
            inds = np.where(np.isnan(numeric_matrix))
            numeric_matrix[inds] = np.take(col_means, inds[1])

            # Compute min/max/mean for normalization
            min_vals = numeric_matrix.min(axis=0)
            max_vals = numeric_matrix.max(axis=0)
            mean_vals = numeric_matrix.mean(axis=0)

            # Save numeric stats for query-time normalization
            self.numeric_stats[label] = {
                'props': numeric_props,
                'min': min_vals.tolist(),
                'max': max_vals.tolist(),
                'mean': mean_vals.tolist()
            }

            # Min-max normalize numeric features to [0,1]
            denom = (max_vals - min_vals)
            denom[denom == 0] = 1.0
            numeric_norm = (numeric_matrix - min_vals) / denom

            print(f" Numeric matrix: {numeric_norm.shape}")

            # Combine TEXT + NUMERIC into one vector for semantic+numeric search
            combined_matrix = np.hstack([text_matrix, numeric_norm.astype('float32')])
            print(f" Combined (text+numeric) matrix: {combined_matrix.shape}")

            # Normalize combined for cosine similarity
            faiss.normalize_L2(combined_matrix)

        # Build FAISS indices
        print(" Building indices...")
        
        # Index 1: Node2Vec 
        self.faiss_node2vec[label] = faiss.IndexFlatIP(self.NODE2VEC_DIM)
        self.faiss_node2vec[label].add(node2vec_matrix)
        print(f"    Node2Vec index: {self.faiss_node2vec[label].ntotal} vectors")
        
        # Index 2: Text
        self.faiss_text[label] = faiss.IndexFlatIP(self.TEXT_DIM)
        self.faiss_text[label].add(text_matrix)
        print(f"    Text index: {self.faiss_text[label].ntotal} vectors")
        
        # Index 3: Combined (if available)
        if combined_matrix is not None:
            combined_dim = combined_matrix.shape[1]
            self.faiss_combined[label] = faiss.IndexFlatIP(combined_dim)
            self.faiss_combined[label].add(combined_matrix)
            print(f"    Combined (text+numeric) index: {self.faiss_combined[label].ntotal} vectors")
        # Building finished; do not save files here. Use `save_faiss_indices` to persist indices to disk.
        return True

    def save_faiss_indices(self, label):
        """Persist FAISS indices and mapping/numeric stats for a label to disk."""
        print(f"\n Saving FAISS indices for {label}...")
        try:
            # File paths
            node2vec_path = os.path.join(self.FAISS_DIR, f"{label}_node2vec.index")
            text_path = os.path.join(self.FAISS_DIR, f"{label}_text.index")
            combined_path = os.path.join(self.FAISS_DIR, f"{label}_combined.index")
            mapping_path = os.path.join(self.FAISS_DIR, f"{label}_mapping.pkl")

            # Save indices
            if label not in self.faiss_node2vec or label not in self.faiss_text:
                print(f"  Missing in-memory FAISS index for {label}; cannot save.")
                return False

            faiss.write_index(self.faiss_node2vec[label], node2vec_path)
            print(f"  Node2Vec saved to {node2vec_path}")

            faiss.write_index(self.faiss_text[label], text_path)
            print(f"  Text saved to {text_path}")

            if label in self.faiss_combined:
                faiss.write_index(self.faiss_combined[label], combined_path)
                print(f"  Combined index saved to {combined_path}")

            # Save mappings + numeric stats
            with open(mapping_path, 'wb') as f:
                pickle.dump({
                    'node_id_to_idx': self.node_id_to_idx[label],
                    'idx_to_node_id': self.idx_to_node_id[label],
                    'numeric_props': self.numeric_props.get(label, []),
                    'numeric_stats': self.numeric_stats.get(label, {})
                }, f)
            print(f"  Mappings & numeric stats saved to {mapping_path}")

            # Save text->node2vec mapper if present for this label
            try:
                mappings2 = {}
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        mappings2 = pickle.load(f)
                if label in self.text2node_W:
                    mappings2['text2node_W'] = self.text2node_W[label]
                    with open(mapping_path, 'wb') as f:
                        pickle.dump(mappings2, f)
                    print(f"  text2node mapper saved to {mapping_path}")
            except Exception as e:
                print(f"  Failed to save text2node mapper: {e}")

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
            print(f"  Node2Vec loaded ({self.faiss_node2vec[label].ntotal} vectors)")
            
            self.faiss_text[label] = faiss.read_index(text_path)
            print(f"  Text loaded ({self.faiss_text[label].ntotal} vectors)")

            # Try to load combined index if it exists
            if os.path.exists(os.path.join(self.FAISS_DIR, f"{label}_combined.index")):
                combined_path = os.path.join(self.FAISS_DIR, f"{label}_combined.index")
                self.faiss_combined[label] = faiss.read_index(combined_path)
                print(f"  Combined loaded ({self.faiss_combined[label].ntotal} vectors)")
            
            # Load mappings + numeric stats
            with open(mapping_path, 'rb') as f:
                mappings = pickle.load(f)
            self.node_id_to_idx[label] = mappings['node_id_to_idx']
            self.idx_to_node_id[label] = mappings['idx_to_node_id']
            self.numeric_props[label] = mappings.get('numeric_props', [])
            self.numeric_stats[label] = mappings.get('numeric_stats', {})
            # Load text->node2vec mapper if stored
            if 'text2node_W' in mappings:
                self.text2node_W[label] = mappings['text2node_W']
                print(f"  text2node mapper loaded for {label}")
            print(f"  Mappings loaded ({len(self.node_id_to_idx[label])} nodes)")
            
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

    def get_numeric_properties(self, label, sample_size=100):
        """
        Auto-detect numeric properties for a label by sampling nodes.
        Returns a list of property names that appear numeric (int/float).
        """
        props = self.get_node_properties(label)
        if not props:
            return []

        # Exclude pricing properties (no pricing data should be used)
        exclude_keywords = ['price', 'cost']

        # Sample values for properties to check types
        sample_query = f"""
        MATCH (n:{label})
        RETURN n LIMIT {sample_size}
        """
        numeric_props = set()
        with self.driver.session() as session:
            res = session.run(sample_query)
            for record in res:
                node = record['n']
                for prop in props:
                    # Skip properties that look like pricing
                    if any(k in prop.lower() for k in exclude_keywords):
                        continue
                    if prop in node and node[prop] is not None:
                        val = node[prop]
                        if isinstance(val, (int, float)):
                            numeric_props.add(prop)
        # Final safety filter
        numeric_props = {p for p in numeric_props if not any(k in p.lower() for k in exclude_keywords)}
        return list(numeric_props)
    
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

    def generate_text_embeddings(self, label, batch_size=1000):
        """
        Generate text embeddings for all nodes of a given label
        Automatically combines relevant text properties
        """
        print(f"\n Generating text embeddings for {label} nodes...")
        
        # Get text properties
        text_props = self.get_text_properties(label)
        primary_key = self.get_primary_key(label)
        
        if not text_props:
            print(f"  No text properties found for {label}")
            return False
        
        print(f" Using properties: {text_props}")
        print(f" Primary key: {primary_key}")
        
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
            print(f"No {label} nodes found")
            return False
        
        print(f"Processing {len(records)} {label} nodes...")
        
        embeddings_to_save = []
        
        #using batching to prevent memory issues [reiews runs out of memory ]   
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            
            # Create text representations
            node_texts = []
            node_ids = []
            
            for record in batch:
                node_id = record['node_id']
                
                # Combine all text properties
                text_parts = []
                for prop in text_props:
                    value = record.get(prop)
                    if value:
                        # Clean and add property
                        text_parts.append(f"{prop}: {value}")
                
                if not text_parts:
                    text_parts.append(f"{label} node")
                
                combined_text = ". ".join(text_parts)
                
                node_texts.append(combined_text)
                node_ids.append(node_id)
            
            # Generate embeddings
            batch_embeddings = self.text_model.encode(
                node_texts,
                show_progress_bar=len(node_texts) > 100,
                batch_size=64
            )
            
            # Prepare for Neo4j
            for node_id, embedding in zip(node_ids, batch_embeddings):
                embeddings_to_save.append({
                    'node_id': node_id,
                    'embedding': embedding.tolist()
                })
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {i + len(batch)}/{len(records)}...")
        
        # Save to Neo4j in batches (important for large datasets!)
        print(f"   Saving {len(embeddings_to_save)} text embeddings to Neo4j for label '{label}'...")
        print(f"   Primary key: {primary_key}")
        print(f"   Batch size: 1000")
        
        save_batch_size = 1000  # Save 1000 at a time to speed up
        total_saved = 0
        failed_batches = 0
        
        for i in range(0, len(embeddings_to_save), save_batch_size):
            batch = embeddings_to_save[i:i+save_batch_size]
            
            try:
                with self.driver.session() as session:
                    result = session.run(f"""
                        UNWIND $embeddings as emb
                        MATCH (n:{label} {{{primary_key}: emb.node_id}})
                        SET n.{self.TEXT_EMBEDDING_PROP} = emb.embedding
                        RETURN count(n) as updated
                    """, embeddings=batch)
                    updated = result.single()['updated']
                    total_saved += updated
                    
                    if updated < len(batch):
                        print(f"       Batch {i//save_batch_size}: Only {updated}/{len(batch)} matched!")
            except Exception as e:
                print(f"      Batch {i//save_batch_size} failed: {e}")
                failed_batches += 1
                continue
            
            if (i // save_batch_size + 1) % 10 == 0:
                print(f"     Progress: {total_saved}/{len(embeddings_to_save)} saved...")
        
        print(f"  Completed: {total_saved}/{len(embeddings_to_save)} saved, {failed_batches} batches failed")
        
        # VERIFY embeddings were actually saved
        print(f"   Verifying embeddings in database...")
        try:
            with self.driver.session() as session:
                verify = session.run(f"""
                    MATCH (n:{label})
                    WHERE n.{self.TEXT_EMBEDDING_PROP} IS NOT NULL
                    RETURN count(n) as count
                """)
                actual_count = verify.single()['count']
            
            print(f"   Verification result: {actual_count} nodes with embeddings")
            
            if actual_count == 0:
                print(f"   WARNING: No embeddings found in database after save!")
                print(f"   Saved {total_saved} but found {actual_count}")
                print(f"   This suggests a primary key mismatch or transaction issue")
                
                # Debug: Show sample node
                with self.driver.session() as session:
                    sample = session.run(f"""
                        MATCH (n:{label})
                        RETURN n.{primary_key} as key LIMIT 1
                    """)
                    sample_record = sample.single()
                    if sample_record:
                        print(f"   Sample {label} node has {primary_key} = {sample_record['key']}")
                
                return False
            elif actual_count < total_saved:
                print(f"    Only {actual_count}/{total_saved} embeddings verified")
                return True  # Still continue, partial success
            else:
                print(f"   Success: All {actual_count} embeddings verified in database")
                return True
        except Exception as e:
            print(f"   Verification failed: {e}")
            return False
        
    
    # ==================== SEARCH METHODS ====================
    
    def find_similar_hotels(self, hotel_name, top_k=5):
        """Find similar hotels using Node2Vec embeddings"""
        print(f"\n Searching similar hotels to: '{hotel_name}'")

        label = 'Hotel'

        # Ensure Hotel index is loaded
        if label not in self.faiss_node2vec:
            print(f"  No Hotel index loaded. Attempting to load...")
            if not self.load_faiss_indices(label):
                print(f"  Failed to load Hotel index. Run setup first.")
                return []

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
        if hotel_id not in self.node_id_to_idx.get(label, {}):
            print(f"  Hotel not in index")
            return []

        idx = self.node_id_to_idx[label][hotel_id]

        # Get Node2Vec embedding
        query_vector = self.faiss_node2vec[label].reconstruct(idx).reshape(1, -1)

        # Search Node2Vec index
        distances, indices = self.faiss_node2vec[label].search(query_vector, top_k + 1)

        # Get results (skip query hotel)
        similar_hotel_ids = []
        scores = []

        for i, score in zip(indices[0], distances[0]):
            result_hotel_id = self.idx_to_node_id[label][i]
            if result_hotel_id == hotel_id:
                continue
            similar_hotel_ids.append(result_hotel_id)
            scores.append(float(score))
            if len(similar_hotel_ids) >= top_k:
                break

        # Enrich with details
        return self._enrich_results(label, similar_hotel_ids, scores, "Node2Vec similarity")
    
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
    
    def _parse_numeric_from_query(self, label, query_text):
        """
        Build a normalized numeric vector from free-text query using simple heuristics.
        - Looks for numbers in the query and tries to map them to numeric properties
        - If no value found for a property, uses the column mean (from numeric_stats)
        """
        props = self.numeric_props.get(label, [])
        stats = self.numeric_stats.get(label, {})
        if not props or not stats:
            return None  # No numeric info available

        # Extract numbers from query
        found_numbers = [float(x) for x in re.findall(r"\d+\.?\d*", query_text)]
        normalized = []

        for i, prop in enumerate(props):
            prop_lower = prop.lower()
            chosen_val = None
            ql = query_text.lower()

            # Derive keyword set from property name and common synonyms
            tokens = re.findall(r"[a-z]+", prop_lower)
            keywords = set(tokens)
            synonyms = {
                'star': ['star', 'stars', 'rating'],
                'cleanliness': ['clean', 'cleanliness'],
                'comfort': ['comfort'],
                'facilities': ['facility', 'facilities'],
                'location': ['location'],
                'staff': ['staff'],
                'value': ['value', 'value_for_money', 'value_for'],
                'overall': ['overall'],
                'score': ['score', 'scored', 'rating']
            }
            for t in tokens:
                if t in synonyms:
                    keywords.update(synonyms[t])

            # Try to find a number directly adjacent to a matching keyword
            for kw in sorted(keywords, key=lambda x: -len(x)):
                if kw in ql:
                    # look for patterns like 'kw 4' or '4 kw'
                    m = re.search(rf"{kw}\W*(\d+\.?\d*)", ql)
                    if not m:
                        m = re.search(rf"(\d+\.?\d*)\W*{kw}", ql)
                    if m:
                        chosen_val = float(m.group(1))
                        break
                    else:
                        # If keyword present and numbers are in query, take first number as likely target
                        if found_numbers:
                            chosen_val = found_numbers[0]
                            break

            # If no keyword match but the query includes exactly one number, use it as a fallback
            if chosen_val is None and len(found_numbers) == 1:
                chosen_val = found_numbers[0]

            # Normalize using stats
            min_vals = np.array(stats.get('min', []), dtype='float32')
            max_vals = np.array(stats.get('max', []), dtype='float32')
            mean_vals = np.array(stats.get('mean', []), dtype='float32')

            if chosen_val is None:
                # use mean
                if mean_vals.size > i:
                    normalized.append(float(mean_vals[i]))
                else:
                    normalized.append(0.0)
            else:
                # scale chosen_val using min-max
                if min_vals.size > i and max_vals.size > i:
                    mn = float(min_vals[i])
                    mx = float(max_vals[i])
                    if mx - mn == 0:
                        normalized.append(float(0.0))
                    else:
                        normalized.append((float(chosen_val) - mn) / (mx - mn))
                else:
                    normalized.append(float(chosen_val))

        return np.array(normalized, dtype='float32').reshape(1, -1)

    # ==================== TEXT -> Node2Vec Mapping ====================
    def train_text2node2vec(self, label, reg=1e-3):
        """Train a linear mapper W so that text_embedding @ W ~= node2vec_embedding.
        Uses ridge regression (closed-form) for speed; saves W to mapping file.
        Returns True on success, False otherwise.
        """
        print(f"\n Training text->node2vec mapper for {label}...")
        primary_key = self.get_primary_key(label)

        query = f"""
        MATCH (n:{label})
        WHERE n.{self.TEXT_EMBEDDING_PROP} IS NOT NULL AND n.{self.NODE2VEC_PROP} IS NOT NULL
        RETURN n.{primary_key} as node_id, n.{self.TEXT_EMBEDDING_PROP} as text_embedding, n.{self.NODE2VEC_PROP} as node2vec_embedding
        ORDER BY node_id
        """
        with self.driver.session() as session:
            result = session.run(query)
            records = result.data()

        if not records or len(records) < 10:
            print(f"  Not enough nodes with both embeddings ({len(records)} found); need >=10")
            return False

        X = np.vstack([np.array(r['text_embedding'], dtype='float32') for r in records])  # (N, text_dim)
        Y = np.vstack([np.array(r['node2vec_embedding'], dtype='float32') for r in records])  # (N, node2vec_dim)

        # Solve ridge: W = (X^T X + reg I)^{-1} X^T Y
        XtX = X.T @ X
        d = XtX.shape[0]
        XtX_reg = XtX + reg * np.eye(d, dtype='float32')
        try:
            W = np.linalg.solve(XtX_reg, X.T @ Y)  # shape (text_dim, node2vec_dim)
        except Exception as e:
            print(f"  Failed to compute mapper: {e}")
            return False

        self.text2node_W[label] = W.astype('float32')

        # Persist into mapping file
        mapping_path = os.path.join(self.FAISS_DIR, f"{label}_mapping.pkl")
        try:
            mappings = {}
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
            mappings['text2node_W'] = self.text2node_W[label]
            with open(mapping_path, 'wb') as f:
                pickle.dump(mappings, f)
            print(f"  text2node mapper saved to {mapping_path}")
        except Exception as e:
            print(f"  Failed to save mapper: {e}")
            return False

        print(f"  Mapper trained using {X.shape[0]} samples; W shape: {W.shape}")
        return True

    def load_text2node2vec(self, label):
        """Load text->node2vec mapper for label from mapping file (if present)."""
        if label in self.text2node_W:
            return True
        mapping_path = os.path.join(self.FAISS_DIR, f"{label}_mapping.pkl")
        if not os.path.exists(mapping_path):
            return False
        try:
            with open(mapping_path, 'rb') as f:
                mappings = pickle.load(f)
            if 'text2node_W' in mappings:
                self.text2node_W[label] = np.array(mappings['text2node_W'], dtype='float32')
                print(f"  Loaded text2node mapper for {label}")
                return True
            return False
        except Exception as e:
            print(f"  Failed to load mapper: {e}")
            return False

    def search_by_query_node2vec(self, query_text, label=None, top_k=5, weight=1.0):
        """Map query text into node2vec space and search Node2Vec FAISS index(es).
        If label is None, searches across all labels that have both W and a node2vec index.
        Auto-creates missing mappers if indices exist.
        """
        print(f"\n Mapping query to Node2Vec space and searching: '{query_text}'")
        q_text_emb = self.text_model.encode([query_text])[0].astype('float32').reshape(1, -1)

        labels_to_search = []
        if label:
            labels_to_search = [label]
        else:
            # If no indices loaded, try to get labels from Neo4j
            if not self.faiss_node2vec:
                labels_to_search = self.get_node_labels()
            else:
                labels_to_search = list(self.faiss_node2vec.keys())

        results = []
        for lbl in labels_to_search:
            # Step 1: Ensure we have a node2vec index
            if lbl not in self.faiss_node2vec:
                print(f"  Loading FAISS indices for {lbl}...")
                self.load_faiss_indices(lbl)
            if lbl not in self.faiss_node2vec:
                print(f"  No Node2Vec index for {lbl}, skipping")
                continue

            # Step 2: Ensure we have the text->node2vec mapper
            if lbl not in self.text2node_W:
                print(f"  Loading text2node mapper for {lbl}...")
                self.load_text2node2vec(lbl)

            # Step 3: If still no mapper, try to create it
            if lbl not in self.text2node_W:
                print(f"  No mapper found for {lbl}, attempting to train...")
                if not self.train_text2node2vec(lbl):
                    print(f"  Failed to create mapper for {lbl}, skipping")
                    continue
                print(f"  Mapper created successfully for {lbl}")

            W = self.text2node_W[lbl]
            # Project into node2vec space
            try:
                y = q_text_emb @ W  # (1, node2vec_dim)
            except Exception as e:
                print(f"  Projection failed for {lbl}: {e}")
                continue

            if weight != 1.0:
                y = y * float(weight)

            y = y.astype('float32')
            faiss.normalize_L2(y)

            dists, idxs = self.faiss_node2vec[lbl].search(y, top_k)
            node_ids = [self.idx_to_node_id[lbl][idx] for idx in idxs[0]]
            scores = dists[0].tolist()
            enriched = self._enrich_results(lbl, node_ids, scores, 'Mapped text->Node2Vec')
            results.extend(enriched)

        # Sort aggregated results
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return results[:top_k]

    def search_by_query(self, query_text, top_k=5):
        """Search across all available labels using the free-text query.
        Automatically loads indices if none are loaded and aggregates/ranks results across labels.
        """
        print(f"\n Searching by query: '{query_text}'")

        # Build query text embedding once
        query_embedding = self.text_model.encode([query_text])[0]
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Ensure we have at least some indices loaded; if none, attempt to load indices for all labels
        labels_to_search = [l for l in set(list(self.faiss_combined.keys()) + list(self.faiss_text.keys()))]
        if not labels_to_search:
            print("  No indices loaded; attempting to load indices for all labels on disk")
            all_labels = self.get_node_labels()
            for lbl in all_labels:
                try:
                    self.load_faiss_indices(lbl)
                except Exception:
                    continue
            labels_to_search = [l for l in set(list(self.faiss_combined.keys()) + list(self.faiss_text.keys()))]

        if not labels_to_search:
            print("  No indices available to search")
            return []

        results = []

        for label in labels_to_search:
            # If combined index exists, prefer it
            if label in self.faiss_combined and label in self.numeric_props and label in self.numeric_stats:
                num_vec = self._parse_numeric_from_query(label, query_text)
                if num_vec is None:
                    # Build numeric vector as column means if no explicit numeric info
                    mean_vals = np.array(self.numeric_stats[label]['mean'], dtype='float32')
                    min_vals = np.array(self.numeric_stats[label]['min'], dtype='float32')
                    max_vals = np.array(self.numeric_stats[label]['max'], dtype='float32')
                    denom = (max_vals - min_vals)
                    denom[denom == 0] = 1.0
                    num_vec = ((mean_vals - min_vals) / denom).astype('float32').reshape(1, -1)

                combined_query = np.hstack([query_embedding, num_vec])
                faiss.normalize_L2(combined_query)
                dists, idxs = self.faiss_combined[label].search(combined_query, top_k)
                node_ids = [self.idx_to_node_id[label][idx] for idx in idxs[0]]
                scores = dists[0].tolist()
                enriched = self._enrich_results(label, node_ids, scores, "Semantic + numeric match")
                results.extend(enriched)
                continue

            # Otherwise fall back to text-only search if available
            if label not in self.faiss_text:
                continue

            # Normalize and search
            fe = query_embedding.copy()
            faiss.normalize_L2(fe)
            dists, idxs = self.faiss_text[label].search(fe, top_k)
            node_ids = [self.idx_to_node_id[label][idx] for idx in idxs[0]]
            scores = dists[0].tolist()
            enriched = self._enrich_results(label, node_ids, scores, "Semantic text match")
            results.extend(enriched)

        # Sort aggregated results by score (descending) and return top_k overall
        results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return results[:top_k]
    
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
                node['label'] = label
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
                    # Use hotel-specific method for richer embeddings (includes reviews)
                    if label == 'Hotel':
                        if not self.generate_hotel_text_embeddings():
                            print(f"  Skipping {label} - failed to generate embeddings")
                            continue
                    else:
                        if not self.generate_text_embeddings(label):
                            print(f"  Skipping {label} - no text properties")
                            continue
                else:
                    print(f"  Found {count} {label} nodes with text embeddings")
                
                # Step 2: Build FAISS indices
                if not self.build_faiss_indices(label):
                    continue
                
                # Step 3: Save indices
                if not self.save_faiss_indices(label):
                    continue
                
                success_count += 1
                
            except Exception as e:
                print(f"  Error processing {label}: {e}")
                import traceback
                traceback.print_exc()
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
    parser.add_argument('--map-query', type=str, help='Map query text to Node2Vec and search (label optional)')
    parser.add_argument('--train-mapper', type=str, help='Train text->Node2Vec mapper for a given label')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    try:
        # Initialize
        print("Connecting to Neo4j...")
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
            
            # Query search (scoped to one label)
            elif args.query:
                results = retriever.search_by_query(
                    args.query,
                    top_k=args.top_k
                )
                print_results(results, f"{args.label} query results")

        # Global query (no label specified)
        elif args.query:
            # Attempt to load indices for all known labels (ignore failures)
            print('\n Loading available indices for global search...')
            all_labels = retriever.get_node_labels()
            for lbl in all_labels:
                try:
                    retriever.load_faiss_indices(lbl)
                except Exception:
                    continue

            results = retriever.search_by_query(args.query, top_k=args.top_k)
            print_results(results, "Global query results")

        # Train mapper if requested
        if args.train_mapper:
            print(f"\n Training text->Node2Vec mapper for label: {args.train_mapper}")
            ok = retriever.train_text2node2vec(args.train_mapper)
            print('Train mapper returned', ok)

        # Mapped query: map text -> node2vec and search
        if args.map_query:
            # If label specified, try to load that index; otherwise load all indices
            if args.label:
                if not retriever.load_faiss_indices(args.label):
                    print(f"\n Run --setup first for label {args.label}!")
                results = retriever.search_by_query_node2vec(args.map_query, label=args.label, top_k=args.top_k)
                print_results(results, f"Mapped Node2Vec query results (label={args.label})")
            else:
                print('\n Loading indices for mapped global search...')
                all_labels = retriever.get_node_labels()
                for lbl in all_labels:
                    try:
                        retriever.load_faiss_indices(lbl)
                    except Exception:
                        continue
                results = retriever.search_by_query_node2vec(args.map_query, label=None, top_k=args.top_k)
                print_results(results, "Mapped Node2Vec global query results")
        
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
            print("\n# Train text->Node2Vec mapper for Hotel")
            print("python generalized_dual_retriever.py --train-mapper Hotel")
            print("\n# Map query to Node2Vec and search (global)")
            print("python generalized_dual_retriever.py --map-query '3 star cleanliness' --top-k 5")
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
