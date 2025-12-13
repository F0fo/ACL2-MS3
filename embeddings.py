

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from db_manager import DBManager


class EmbeddingRetriever:
    
    # Configuration
    GRAPH_NAME = "travelGraph"
    EMBEDDING_PROP = "node2vecEmbedding"
    VECTOR_INDEX = "hotel_node2vec_index"
    
    def __init__(self, driver):
        self.driver = driver
    
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
                print("  ✓ Dropped existing graph projection")
            except ClientError as e:
                if "not found" not in str(e).lower():
                    print(f"Warning: Could not drop graph: {e}")
            
            # Create new graph projection
            try:
                result = session.run(create_query)
                data = result.data()[0]
                print(f"  ✓ Graph created: {data['nodeCount']} nodes, {data['relationshipCount']} relationships")
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
                print(f"  ✓ Node2Vec complete:")
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
                print(f"  ✓ Vector index '{self.VECTOR_INDEX}' created")
            except ClientError as e:
                print(f" Error creating vector index: {e}")
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
    
    def find_similar_hotels(self, hotel_name, top_k=5):
        """Find similar hotels based on embeddings
        
        Args:
            hotel_name: Name of the hotel to find similar hotels for
            top_k: Number of similar hotels to return
            
        Returns:
            List of dictionaries with hotel names and similarity scores
        """
        query = f"""
        MATCH (h:Hotel {{name: $hotel}})
        WHERE h.{self.EMBEDDING_PROP} IS NOT NULL
        CALL db.index.vector.queryNodes(
          '{self.VECTOR_INDEX}',
          $k + 1,
          h.{self.EMBEDDING_PROP}
        )
        YIELD node, score
        WHERE node.name <> $hotel
        RETURN node.name AS hotel, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $k
        """
        
        with self.driver.session() as session:
            try:
                result = session.run(query, hotel=hotel_name, k=top_k)
                return result.data()
            except Exception as e:
                print(f" Error finding similar hotels: {e}")
                return []
    
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
        query = """
        MATCH (h:Hotel)
        WHERE h.node2vecEmbedding IS NOT NULL
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
        """Complete setup process: create graph, run node2vec, create index"""
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
            
            # Step 4: Create vector index
            self.create_vector_index()
            
            print("\n" + "="*60)
            print("EMBEDDING SETUP COMPLETE")
            print("="*60)
            return True
            
        except Exception as e:
            print(f"\nSetup failed: {e}")
            return False


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hotel Similarity using Node2Vec Embeddings')
    parser.add_argument('--setup', action='store_true', help='Setup embeddings (create graph, run node2vec, create index)')
    parser.add_argument('--hotel', type=str, help='Hotel name to find similar hotels for')
    parser.add_argument('--hotel-id', type=int, help='Hotel ID to find similar hotels for')
    parser.add_argument('--top-k', type=int, default=5, help='Number of similar hotels to return (default: 5)')
    parser.add_argument('--random', action='store_true', help='Find similar hotels for a random hotel')
    
    args = parser.parse_args()
    
    try:
        # Initialize database manager
        print(" Connecting to Neo4j...")
        db_manager = DBManager()
        
        # Initialize embedding retriever
        retriever = EmbeddingRetriever(db_manager.driver)
        
        # Setup embeddings if requested
        if args.setup:
            success = retriever.setup_embeddings()
            if not success:
                db_manager.close()
                exit(1)
        
        # Find similar hotels
        if args.hotel or args.hotel_id or args.random:
            # Verify embeddings exist
            if not retriever.verify_embeddings_exist():
                print("\n  No embeddings found. Run with --setup first:")
                print("   python embedding_retriever.py --setup")
                db_manager.close()
                exit(1)
            
            # Get hotel name
            if args.random:
                hotel_name, hotel_id = retriever.get_random_hotel()
                if not hotel_name:
                    print(" No hotels with embeddings found")
                    db_manager.close()
                    exit(1)
                print(f"\n Random hotel selected: {hotel_name} (ID: {hotel_id})")
            elif args.hotel_id:
                hotel_name = retriever.get_hotel_by_id(args.hotel_id)
                if not hotel_name:
                    print(f" No hotel found with ID: {args.hotel_id}")
                    db_manager.close()
                    exit(1)
            else:
                hotel_name = args.hotel
            
            # Find similar hotels
            print(f"\n Finding similar hotels to: '{hotel_name}'")
            print(f"   (Top {args.top_k} results)\n")
            
            similar_hotels = retriever.find_similar_hotels(hotel_name, top_k=args.top_k)
            
            if similar_hotels:
                print(" Similar Hotels:")
                print("-" * 60)
                for i, record in enumerate(similar_hotels, 1):
                    print(f"{i}. {record['hotel']}")
                    print(f"   Similarity Score: {record['score']:.4f}")
                    print(f"   Hotel ID: {record['hotel_id']}")
                    print()
            else:
                print(f" No similar hotels found for '{hotel_name}'")
                print("   (Hotel may not exist or has no embedding)")
        
        elif not args.setup:
            # No action specified, show help
            parser.print_help()
            print("\nExamples:")
            print("  # Setup embeddings")
            print("  python embedding_retriever.py --setup")
            print()
            print("  # Find similar hotels")
            print("  python embedding_retriever.py --hotel 'Hotel California' --top-k 10")
            print()
            print("  # Find similar hotels for random hotel")
            print("  python embedding_retriever.py --random --top-k 5")
        
        # Close connection
        db_manager.close()
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()