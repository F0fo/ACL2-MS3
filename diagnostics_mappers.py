import faiss
import numpy as np
from numpy.linalg import norm
from embeddings import DualEmbeddingRetriever
from db_manager import DBManager

q = "hotels in paris"

print("Connecting to DB and initializing retriever...")
db = DBManager()
r = DualEmbeddingRetriever(db.driver)

labels = r.get_node_labels()
print("Labels:", labels)

for lbl in labels:
    idx_ok = r.load_faiss_indices(lbl)
    map_ok = r.load_text2node2vec(lbl)
    ntotal = r.faiss_node2vec[lbl].ntotal if lbl in r.faiss_node2vec else 0
    print(f"{lbl}: index_loaded={idx_ok}, mapper_loaded={map_ok}, ntotal={ntotal}")

# Per-label mapped search
print("\nPer-label mapped search results for query:\n", q)
q_emb = r.text_model.encode([q])[0].astype('float32')

for lbl in labels:
    if lbl in r.faiss_node2vec and lbl in r.text2node_W:
        W = r.text2node_W[lbl]
        y = q_emb @ W
        y = y.astype('float32').reshape(1, -1)
        faiss.normalize_L2(y)
        d, idxs = r.faiss_node2vec[lbl].search(y, 3)
        node_ids = [r.idx_to_node_id[lbl][i] for i in idxs[0]]
        print(f"\nLabel={lbl} | top_scores={d[0].tolist()} | node_ids={node_ids}")
    else:
        print(f"\nLabel={lbl} skipped (index or mapper missing)")

print('\nDone')

db.close()
