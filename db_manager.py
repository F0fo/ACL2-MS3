from neo4j import GraphDatabase
from neo4j.exceptions import ClientError


class DBManager:
    """Manages Neo4j database connection + status checks"""
    
    def __init__(self, config_file='config.txt'):
        """Initialize DBManager with config file"""
        self.config = self.read_config(config_file)
        self.driver = self.get_driver()
    
    @staticmethod
    def read_config(filename='config.txt'):
        """Read Neo4j connection details from config file
        
        Expected config.txt format:
        URI=bolt://localhost:7687
        USERNAME=neo4j
        PASSWORD=your_password
        """
        config = {}
        try:
            with open(filename) as c:
                for line in c:
                    if '=' in line:
                        key, value = line.strip().split("=", 1)  # split only on first =
                        config[key] = value
            
            # Validate required keys
            required = ['URI', 'USERNAME', 'PASSWORD']
            missing = [k for k in required if k not in config]
            if missing:
                raise ValueError(f"Missing required config keys: {missing}")
            
            print(f"Config loaded successfully")
            print(f"  URI: {config['URI']}")
            print(f"  Username: {config['USERNAME']}")
            print(f"  Password: {'*' * len(config['PASSWORD'])}")
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file '{filename}' not found")
        except Exception as e:
            raise Exception(f"Error reading config: {e}")
    
    def get_driver(self):
        """Create and return Neo4j driver"""
        try:
            driver = GraphDatabase.driver(
                self.config['URI'],
                auth=(self.config['USERNAME'], self.config['PASSWORD'])
            )
            # Test connection
            driver.verify_connectivity()
            print("✓ Connected to Neo4j successfully")
            return driver
        except Exception as e:
            raise Exception(f"Failed to connect to Neo4j: {e}")
    
    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
            print("✓ Database connection closed")
    
    def clear_db(self):
        """Clear all nodes and relationships from database"""
        with self.driver.session() as session:
            try:
                session.run("MATCH (n) DETACH DELETE n")
                print("✓ Database cleared")
            except Exception as e:
                print(f"✗ Error clearing database: {e}")
                raise
    
    def check_db_status(self):
        """Check current database status - count nodes and relationships"""
        with self.driver.session() as session:
            try:
                # Count nodes by label
                node_result = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                    ORDER BY label
                """)
                node_counts = node_result.data()
                
                # Count relationships by type
                rel_result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY type
                """)
                rel_counts = rel_result.data()
                
                return {
                    'nodes': node_counts,
                    'relationships': rel_counts
                }
            except Exception as e:
                print(f"✗ Error checking database status: {e}")
                return {'nodes': [], 'relationships': []}
    
    def is_db_populated(self):
        """Check if database has any data"""
        with self.driver.session() as session:
            try:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()['count']
                return count > 0
            except Exception as e:
                print(f"✗ Error checking if database is populated: {e}")
                return False
    
    def print_status(self):
        """Print formatted database status"""
        status = self.check_db_status()
        
        print("\n" + "="*60)
        print("DATABASE STATUS")
        print("="*60)
        
        print("\n Nodes:")
        if status['nodes']:
            total_nodes = sum(item['count'] for item in status['nodes'])
            for item in status['nodes']:
                print(f"  • {item['label']:<15} : {item['count']:>6,}")
            print(f"  {'-'*25}")
            print(f"  • {'TOTAL':<15} : {total_nodes:>6,}")
        else:
            print("  No nodes found")
        
        print("\n Relationships:")
        if status['relationships']:
            total_rels = sum(item['count'] for item in status['relationships'])
            for item in status['relationships']:
                print(f"  • {item['type']:<15} : {item['count']:>6,}")
            print(f"  {'-'*25}")
            print(f"  • {'TOTAL':<15} : {total_rels:>6,}")
        else:
            print("  No relationships found")
        
        print("="*60 + "\n")
    
    def get_session(self):
        """Get a new database session (context manager compatible)"""
        return self.driver.session()


# For backwards compatibility with old code
def readConfig(filename='config.txt'):
    """Legacy function - use DBManager.read_config() instead"""
    return DBManager.read_config(filename)


def getDriver(config):
    """Legacy function - use DBManager() instead"""
    driver = GraphDatabase.driver(
        config['URI'],
        auth=(config['USERNAME'], config['PASSWORD'])
    )
    return driver


if __name__ == "__main__":
    """Test the DBManager"""
    try:
        # Initialize manager
        db = DBManager()
        
        # Check status
        db.print_status()
        
        # Close connection
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")