import pandas as pd
from neo4j import GraphDatabase

from create_kg import getDriver, readConfig

class EmbeddingRetriever:

    HF_TOKEN = "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"   
    config = readConfig()
    driver = getDriver(config)


    def get_node_embeddings(session):
        result = session.run("""
        MATCH (n)
        WHERE
            n:Country OR
            n:City OR
            n:Hotel OR
            n:Traveller OR
            n:Review
        RETURN
            id(n) AS nodeId,
            labels(n) AS labels,
            coalesce(n.name, n.review_id, n.hotel_id, n.user_id) AS identifier,
            n.embedding AS embedding;
        """)                    
