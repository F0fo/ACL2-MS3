import os
import logging
from typing import List, Dict, Any, Optional
import requests
import json
import os.path

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_config_file(path: Optional[str] = None) -> Dict[str, str]:
	"""
	Read simple key=value lines from config file. Comments with '#' or ';' are ignored.
	Default path: env CONFIG_FILE or ./config.txt next to this file.
	"""
	if path is None:
		path = os.getenv("CONFIG_FILE") or os.path.join(os.path.dirname(__file__), "config.txt")
	if not os.path.exists(path):
		return {}
	out: Dict[str, str] = {}
	with open(path, "r", encoding="utf-8") as fh:
		for ln in fh:
			ln = ln.strip()
			if not ln or ln.startswith("#") or ln.startswith(";"):
				continue
			if "=" not in ln:
				continue
			k, v = ln.split("=", 1)
			k = k.strip().upper()
			v = v.strip().strip('"').strip("'")
			out[k] = v
	return out


class VectorEmbedder:
	def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
	             text_model: str = "all-MiniLM-L6-v2",
	             use_hf_if_available: bool = True,
	             config: Optional[Dict[str, str]] = None):
		# allow passing config dict or read from file
		if config is None:
			config = read_config_file()
		# Neo4j driver setup
		self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
		self.text_model = text_model
		# detect huggingface token from config file first, then common env var names
		self.hf_token = (config.get("HUGGINGFACE_API_KEY")
		                 or config.get("HUGGINGFACE_API_TOKEN")
		                 or config.get("HUGGINGFACE_TOKEN")
		                 or os.getenv("HUGGINGFACE_API_KEY")
		                 or os.getenv("HUGGINGFACE_API_TOKEN")
		                 or os.getenv("HUGGINGFACE_TOKEN"))
		self.hf_model = config.get("HUGGINGFACE_MODEL") or os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
		self.use_hf = use_hf_if_available and (self.hf_token is not None)
		if self.use_hf:
			logger.info("Using HuggingFace Inference API model=%s", self.hf_model)
		else:
			try:
				# local encoder (default)
				from sentence_transformers import SentenceTransformer
				self.model = SentenceTransformer(self.text_model)
			except Exception:
				SentenceTransformer = None
				raise RuntimeError("SentenceTransformer not available; set HUGGINGFACE_API_KEY in config.txt or install sentence-transformers")

	def close(self):
		self.driver.close()

	def _fetch_hotels(self, batch_size: int = 1000):
		# fetch hotels and related nodes: city, country, reviews, travellers
		with self.driver.session() as session:
			result = session.run(
				"""
				MATCH (h:Hotel)
				OPTIONAL MATCH (h)-[:IN_CITY]->(city:City)
				OPTIONAL MATCH (city)-[:IN_COUNTRY]->(country:Country)
				OPTIONAL MATCH (h)-[:HAS_REVIEW]->(rev:Review)
				OPTIONAL MATCH (rev)-[:BY_TRAVELLER]->(t1:Traveller)
				OPTIONAL MATCH (h)-[:HAS_TRAVELLER]->(t2:Traveller)
				RETURN id(h) AS id,
					   h.name AS name,
					   h.location AS location,
					   h.amenities AS amenities,
					   h.avg_rating AS avg_rating,
					   collect(DISTINCT rev.text) AS reviews,
					   city.name AS city,
					   country.name AS country,
					   collect(DISTINCT t1.name) AS travellers_from_reviews,
					   collect(DISTINCT t2.name) AS travellers_direct
				"""
			)
			for record in result:
				# combine traveller lists into one and filter nulls
				t1 = record.get("travellers_from_reviews") or []
				t2 = record.get("travellers_direct") or []
				travellers = [str(x) for x in (t1 + t2) if x is not None]
				yield {
					"id": record["id"],
					"name": record.get("name"),
					"location": record.get("location"),
					"amenities": record.get("amenities"),
					"avg_rating": record.get("avg_rating"),
					"reviews": record.get("reviews") or [],
					"city": record.get("city"),
					"country": record.get("country"),
					"travellers": travellers,
				}

	def _combine_features(self, node: Dict[str, Any]) -> str:
		parts = []
		if node.get("name"):
			parts.append(f"Name: {node['name']}")
		# prefer explicit city/country when available
		city = node.get("city")
		country = node.get("country")
		if city or country:
			loc = ", ".join(p for p in (city, country) if p)
			parts.append(f"Location: {loc}")
		else:
			if node.get("location"):
				parts.append(f"Location: {node['location']}")
		# amenities could be list or comma string
		amen = node.get("amenities")
		if amen:
			if isinstance(amen, (list, tuple)):
				amen = ", ".join(str(a) for a in amen)
			parts.append(f"Amenities: {amen}")
		if node.get("avg_rating") is not None:
			parts.append(f"Average rating: {node['avg_rating']}")
		# reviews could be list of strings or single string
		revs = node.get("reviews") or []
		if revs:
			# keep a few reviews to limit length
			revs_text = " | ".join(str(r) for r in revs[:5])
			parts.append(f"Reviews: {revs_text}")
			parts.append(f"Review count: {len(revs)}")
		# travellers: include names or roles if available
		travs = node.get("travellers") or []
		if travs:
			travs_text = ", ".join(str(t) for t in travs[:5])
			parts.append(f"Travellers: {travs_text}")
			parts.append(f"Traveller count: {len(travs)}")
		return ". ".join(parts)

	def _encode_texts(self, texts: List[str]) -> List[List[float]]:
		if self.use_hf:
			# Call Hugging Face feature-extraction pipeline in batch
			url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.hf_model}"
			headers = {"Authorization": f"Bearer {self.hf_token}"}
			try:
				resp = requests.post(url, headers=headers, json=texts, timeout=60)
				if resp.status_code != 200:
					logger.error("HF inference error %s: %s", resp.status_code, resp.text)
					raise RuntimeError(f"HuggingFace API error: {resp.status_code}")
				data = resp.json()
				# HF returns list-of-embeddings corresponding to inputs
				# Each item may be nested lists if model returns token-level vectors; handle flattening to a single vector per text.
				processed = []
				for item in data:
					if isinstance(item, list) and item and isinstance(item[0], list):
						# mean-pool token vectors
						import math
						vec_len = len(item[0])
						mean_vec = [0.0] * vec_len
						for token_vec in item:
							for i, v in enumerate(token_vec):
								mean_vec[i] += v
						n = len(item)
						if n > 0:
							mean_vec = [x / n for x in mean_vec]
						processed.append(mean_vec)
					else:
						processed.append(item)
				return processed
			except Exception as e:
				logger.exception("HuggingFace embedding request failed: %s", e)
				raise
		else:
			# SentenceTransformer encode returns numpy arrays
			embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
			return [e.tolist() for e in embs]

	def _write_embeddings(self, rows: List[Dict[str, Any]]):
		# rows: list of {id: int, embedding: List[float]}
		with self.driver.session() as session:
			# Use UNWIND for batch update
			session.run(
				"""
				UNWIND $rows AS r
				MATCH (n) WHERE id(n)=r.id
				SET n.embedding = r.embedding
				""",
				rows=rows,
			)

	def generate_and_write_embeddings(self, batch_size: int = 256, normalize: bool = True, append_avg_rating: bool = True):
		# Iterate hotels, build texts, encode in batches, optionally append normalized avg_rating, and write back.
		buffer = []
		meta = []
		for node in self._fetch_hotels():
			text = self._combine_features(node)
			buffer.append(text)
			meta.append(node)
			if len(buffer) >= batch_size:
				self._process_batch(buffer, meta, normalize, append_avg_rating)
				buffer = []
				meta = []
		if buffer:
			self._process_batch(buffer, meta, normalize, append_avg_rating)
		logger.info("Embedding update complete.")

	def _process_batch(self, texts: List[str], metas: List[Dict[str, Any]], normalize: bool, append_avg_rating: bool):
		embs = self._encode_texts(texts)
		# optionally normalize and append numeric features
		processed_rows = []
		for emb, meta in zip(embs, metas):
			vec = emb
			# append normalized avg_rating as extra float dimension if requested
			if append_avg_rating:
				avg = meta.get("avg_rating")
				if avg is None:
					norm_avg = 0.0
				else:
					try:
						norm_avg = float(avg) / 5.0  # assuming rating out of 5
					except Exception:
						norm_avg = 0.0
				vec = list(vec) + [norm_avg]
			if normalize:
				# L2 normalize
				try:
					import math
					norm = math.sqrt(sum(x * x for x in vec))
					if norm > 0:
						vec = [x / norm for x in vec]
				except Exception:
					pass
			processed_rows.append({"id": meta["id"], "embedding": vec})
		self._write_embeddings(processed_rows)

	def test_first_n(self, n: int = 10, normalize: bool = True, append_avg_rating: bool = True) -> List[Dict[str, Any]]:
		"""
		Fetch first `n` hotels, compute combined feature texts and embeddings,
		apply optional avg_rating append & normalization, and return list of dicts:
		[{"id": id, "text": combined_text, "embedding": [...]}, ...]
		"""
		metas: List[Dict[str, Any]] = []
		texts: List[str] = []
		for i, node in enumerate(self._fetch_hotels()):
			if i >= n:
				break
			metas.append(node)
			texts.append(self._combine_features(node))
		if not texts:
			return []
		embs = self._encode_texts(texts)
		results: List[Dict[str, Any]] = []
		import math
		for emb, meta, text in zip(embs, metas, texts):
			vec = list(emb)
			if append_avg_rating:
				avg = meta.get("avg_rating")
				try:
					norm_avg = float(avg) / 5.0 if avg is not None else 0.0
				except Exception:
					norm_avg = 0.0
				vec = vec + [norm_avg]
			if normalize:
				norm = math.sqrt(sum(x * x for x in vec))
				if norm > 0:
					vec = [x / norm for x in vec]
			results.append({"id": meta["id"], "text": text, "embedding": vec})
		return results


def main():
	# Read credentials from config.txt first, then env vars / defaults
	cfg = read_config_file()
	NEO4J_URI = cfg.get("NEO4J_URI") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
	NEO4J_USER = cfg.get("NEO4J_USER") or os.getenv("NEO4J_USER", "neo4j")
	NEO4J_PASSWORD = cfg.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", "password")
	# Use Hugging Face if token present (config or env)
	USE_HF = (cfg.get("HUGGINGFACE_API_KEY") or cfg.get("HUGGINGFACE_TOKEN") or
	          os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")) is not None
	E = VectorEmbedder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, use_hf_if_available=USE_HF, config=cfg)
	try:
		E.generate_and_write_embeddings(batch_size=256, normalize=True, append_avg_rating=True)
	finally:
		E.close()


def test_first_n_hotels(n: int = 10):
	# convenience helper that reads config.txt like main and returns test result
	cfg = read_config_file()
	NEO4J_URI = cfg.get("NEO4J_URI") or os.getenv("NEO4J_URI", "bolt://localhost:7687")
	NEO4J_USER = cfg.get("NEO4J_USER") or os.getenv("NEO4J_USER", "neo4j")
	NEO4J_PASSWORD = cfg.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD", "password")
	USE_HF = (cfg.get("HUGGINGFACE_API_KEY") or cfg.get("HUGGINGFACE_TOKEN") or
	          os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_TOKEN")) is not None
	E = VectorEmbedder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, use_hf_if_available=USE_HF, config=cfg)
	try:
		return E.test_first_n(n)
	finally:
		E.close()


if __name__ == "__main__":
	# if TEST_FIRST_N set, run the quick test and print JSON; otherwise run main behavior
	if os.getenv("TEST_FIRST_N"):
		n = int(os.getenv("TEST_FIRST_N", "10"))
		res = test_first_n_hotels(n)
		print(json.dumps(res, default=str, indent=2))
	else:
		main()