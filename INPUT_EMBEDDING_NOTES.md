# Input Embedding - Implementation Notes

## Requirement (from project spec)

> **c. Input Embedding (depending on 2.b)**
> Convert the user's text input into a vector representation for semantic similarity search in the embedding-based retrieval approach. Only needed when you implement embedding-based retrieval (section 2.b). Use the same embedding model that was used to create node or feature vector embeddings in your KG.

---

## Current Implementation Status: Partial

### What We Have
- **Node2Vec**: 128-dim graph structure embeddings (random walks)
- **FastRP**: 128-dim fast random projection embeddings
- Both are **graph-based** embeddings that encode nodes based on their position/connections in the graph

### The Problem
Node2Vec and FastRP are **not text encoders**. They:
- Embed nodes based on graph topology
- Cannot directly convert arbitrary text queries into vectors
- Only work with nodes that already exist in the graph

### Current Workaround
1. Extract hotel names from user query (entity extraction)
2. Look up the hotel's pre-computed embedding in FAISS
3. Find similar hotels based on that embedding

This works but is **not truly converting user text input into a vector**.

---

## Options to Fix

### Option 1: Text-to-Graph-Embedding Mapper
Train a model that maps text queries to the graph embedding space.

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

# 1. Encode hotel descriptions with text encoder
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim
hotel_descriptions = ["The Azure Tower in Tokyo, 5-star luxury hotel..."]
text_embeddings = text_encoder.encode(hotel_descriptions)

# 2. Train mapper from text space (384-dim) to graph space (128-dim)
mapper = Ridge()
mapper.fit(text_embeddings, node2vec_embeddings)

# 3. At query time
query = "luxury hotel in Tokyo"
query_text_emb = text_encoder.encode([query])
query_graph_emb = mapper.predict(query_text_emb)  # Now in 128-dim graph space

# 4. Search FAISS with mapped embedding
```

**Pros:** Uses same embedding space as nodes
**Cons:** Requires training, may lose precision in mapping

---

### Option 2: Text Embeddings of Node Properties (Recommended)
Create separate text embeddings from hotel descriptions and use those for query matching.

**Implementation:**
```python
from sentence_transformers import SentenceTransformer
import faiss

# 1. Create text embeddings for each hotel
text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

hotel_texts = []
for hotel in hotels:
    text = f"{hotel['name']} in {hotel['city']}, {hotel['country']}. "
    text += f"{hotel['star_rating']}-star hotel. "
    text += f"Cleanliness: {hotel['avg_cleanliness']}, Comfort: {hotel['avg_comfort']}"
    hotel_texts.append(text)

text_embeddings = text_encoder.encode(hotel_texts)

# 2. Build FAISS index for text embeddings
text_index = faiss.IndexFlatIP(384)
faiss.normalize_L2(text_embeddings)
text_index.add(text_embeddings)

# 3. At query time - encode query with same encoder
query = "clean comfortable hotel in Paris"
query_embedding = text_encoder.encode([query])
faiss.normalize_L2(query_embedding)

# 4. Search text embedding index
D, I = text_index.search(query_embedding, k=5)
```

**Pros:**
- Direct text-to-text similarity
- Same encoder for query and hotels
- Handles natural language queries well

**Cons:**
- Separate from graph embeddings
- Doesn't capture graph structure

---

### Option 3: Accept Current Approach
Document that for graph embeddings, we use entity extraction to find the relevant node, then use its pre-computed embedding.

**Justification:**
- Graph embeddings (Node2Vec/FastRP) are fundamentally different from text embeddings
- Entity extraction bridges the gap between text and graph
- The requirement may be interpreted as: "use embeddings for retrieval" rather than "encode raw text"

**Documentation:**
> For graph-based embeddings, user queries are processed through entity extraction to identify mentioned hotels. The extracted hotel's graph embedding is then used for similarity search. This approach leverages the graph structure while using NLP for query understanding.

---

## Recommendation

**Option 2** is most aligned with the requirement and provides the best user experience:
- Users can type natural language queries
- Semantic matching against hotel properties
- Can be combined with graph embeddings for hybrid approach

**Implementation effort:** ~2-3 hours
- Add SentenceTransformer encoding for hotels
- Build text FAISS index
- Add query encoding in search flow
- Update UI to show text similarity results

---

## Files to Modify

1. `embeddings_retreiver.py` - Add TextEmbeddingRetriever class
2. `chatbot.py` - Integrate text embeddings in retrieval
3. `app.py` - Add option to use text-based search

---

## Decision Needed

- [ ] Implement Option 1 (mapper)
- [ ] Implement Option 2 (text embeddings)
- [ ] Accept Option 3 (current approach)
- [ ] Discuss with team/instructor first
