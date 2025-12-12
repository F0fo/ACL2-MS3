import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#from entity_extractor import extract_entities, get_cypher_params
#from sentence_embedding import embed_query # Uncomment after implementing sentence embedding

# Load pre-trained model and tokenizer for entity extraction
#tokenizer = AutoTokenizer.from_pretrained("intent_model_final")
#model = AutoModelForSequenceClassification.from_pretrained("intent_model_final")


st.title("Hotel Recommendation system")

user_question = st.text_input("Need help with your travel plans? Ask me anything about hotels!")

if user_question:
    #Intent Prediction
    inputs = tokenizer(user_question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()
    confidence = outputs.logits.softmax(dim=1).max().item()

    st.write(f"**Predicted Intent:** {predicted_class} (Confidence: {confidence:.3f})")

    # Entity Extraction
    entities = extract_entities(user_question)
    st.write("**Extracted Entities:**", entities)

    #Embedding - edit after implementing sentence embedding
    #vector = embed_query(user_question)
    #st.write(f"**Embedding Vector:** {vector} (Embedding size: {len(vector)})")