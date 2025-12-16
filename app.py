import streamlit as st
from entity_extractor import extract_entities, get_cypher_params
from chatbot import process_query_for_baseline, call_llm
from db_manager import DBManager
from intent_classifier import classify_intent
from streamlit_agraph import agraph, Node, Edge, Config
from embeddings_retreiver import HotelEmbeddingRetriever
import pandas as pd

# Page config
st.set_page_config(
    page_title="J.A.R.V.I.S. Hotel Concierge",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #1a1a2e 100%);
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #00d4ff;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #888;
        margin: 0.5rem 0 0 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DBManager()


def visualize_neo4j_subgraph(context, params):
    """Visualize graph from Neo4j paths"""
    nodes = []
    edges = []
    node_map = {}

    colors = {"Hotel": "#FF6B6B", "City": "#4ECDC4", "Country": "#45B7D1",
              "Review": "#FFA07A", "Traveller": "#95E1D3"}

    for ctx in context:
        if isinstance(ctx, dict) and 'graph' in ctx:
            g = ctx['graph']
            # Add nodes
            for node in g.get('nodes', []):
                node_id = node['id']
                if node_id not in node_map:
                    label_type = node['labels'][0] if node['labels'] else 'Unknown'
                    props = node['properties']

                    # Create label
                    if 'name' in props:
                        label = props['name']
                    elif 'text' in props:
                        label = f"Review ({props.get('score_overall', 'N/A')})"
                    elif 'user_id' in props:
                        label = f"{props.get('type', 'Traveller')}"
                    else:
                        label = label_type

                    # Create detailed title for hover
                    title_parts = [f"{label_type}: {label}"]
                    for key, val in props.items():
                        if key not in ['name', 'user_id'] and val:
                            title_parts.append(f"{key}: {val}")
                    title = "\n".join(title_parts)

                    size = 30 if label_type == 'Hotel' else 20
                    color = colors.get(label_type, "#888")

                    nodes.append(Node(id=node_id, label=label, size=size, color=color, title=title))
                    node_map[node_id] = {'label': label, 'type': label_type, 'props': props}

            # Add relationships
            for rel in g.get('relationships', []):
                edges.append(Edge(source=rel['start'], target=rel['end'], label=rel['type']))

    if nodes:
        config = Config(width=750, height=500, directed=True, physics=True, hierarchical=False)
        selected = agraph(nodes=nodes, edges=edges, config=config)

        # Show selected node details
        if selected:
            st.subheader("Selected Node Details")
            node_info = node_map.get(selected)
            if node_info:
                st.write(f"**Type:** {node_info['type']}")
                st.write(f"**Label:** {node_info['label']}")
                if node_info['props']:
                    st.write("**Properties:**")
                    st.json(node_info['props'])
    else:
        st.info("No graph data available")


# Header
st.markdown("""
<div class="main-header">
    <h1>J.A.R.V.I.S.</h1>
    <p>Your AI Hotel Concierge</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("### Settings")
    st.markdown("---")

    model_option = st.selectbox(
        "LLM Model",
        options=["Gemma", "Mistral", "LLaMA"],
        index=0,
        help="Select the language model for generating responses"
    )

    embedding_option = st.selectbox(
        "Embedding Model",
        options=["Node2Vec (128-dim)", "FastRP (128-dim)"],
        index=0,
        help="Node2Vec: Random walk-based\nFastRP: Fast random projection"
    )
    embedding_model = 'node2vec' if 'Node2Vec' in embedding_option else 'fastrp'


# Sample questions
st.markdown("##### Quick Test Questions:")
sample_questions = [
    "Recommend me a good hotel in Tokyo",
    "Find hotels in Paris",
    "Tell me about The Azure Tower",
    "Show me reviews for The Golden Oasis",
    "Compare The Azure Tower and L'Etoile Palace",
    "Best hotels for business travelers",
    "Hotels with cleanliness rating above 9",
    "Hotels recommended for seniors",
]

# Initialize session state for selected question
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""

# Create button grid
col1, col2 = st.columns(2)
with col1:
    for i in range(0, 4):
        if st.button(f"{sample_questions[i]}", key=f"q{i}", use_container_width=True):
            st.session_state.selected_question = sample_questions[i]

with col2:
    for i in range(4, 8):
        if st.button(f"{sample_questions[i]}", key=f"q{i}", use_container_width=True):
            st.session_state.selected_question = sample_questions[i]

# Query input
user_input = st.text_input("Or type your own question:", value=st.session_state.selected_question)
user_question = user_input

# Clear selection after use
if user_question and user_question == st.session_state.selected_question:
    st.session_state.selected_question = ""

if user_question:
    # Intent & Entities
    intent_result = classify_intent(user_question)
    entities = extract_entities(user_question)
    params = get_cypher_params(entities)

    st.subheader("Intent & Entities")
    col1, col2 = st.columns(2)
    col1.metric("Intent", intent_result['intent'])
    col2.metric("Confidence", f"{intent_result['confidence']:.2%}")
    if entities:
        st.json(entities)

    # KG Context
    st.subheader("Knowledge Graph Context")
    context = process_query_for_baseline(user_question, st.session_state.db_manager.driver)

    valid_context = [c for c in context if c]
    if valid_context:
        for idx, ctx in enumerate(valid_context, 1):
            with st.expander(f"Context {idx}", expanded=False):
                # Show cypher query
                query_text = ctx.get('query') if isinstance(ctx, dict) else None
                if query_text:
                    st.code(query_text, language='cypher')

                # Render rows (aggregations) or graph
                if isinstance(ctx, dict) and 'rows' in ctx and ctx['rows']:
                    st.dataframe(pd.DataFrame(ctx['rows']), use_container_width=True)
                elif isinstance(ctx, dict) and 'graph' in ctx:
                    st.json({'nodes': len(ctx['graph'].get('nodes', [])), 'relationships': len(ctx['graph'].get('relationships', []))})
                    # Provide raw preview
                    st.write("Sample nodes (up to 5):")
                    sample_nodes = ctx['graph'].get('nodes', [])[:5]
                    st.json(sample_nodes)
                else:
                    st.write(ctx)
    else:
        st.warning("No context retrieved")

    # Graph Visualization
    if valid_context:
        st.subheader("Graph Visualization")
        visualize_neo4j_subgraph(valid_context, params)

    # Embedding Comparison
    st.subheader("Embedding Model Comparison")

    # Extract hotel name from entities for similarity search
    reference_hotel = entities.get('hotels', [None])[0] if entities.get('hotels') else None

    if reference_hotel:
        st.caption(f"Finding hotels similar to **{reference_hotel}** using different embedding models")

        try:
            emb_col1, emb_col2 = st.columns(2)

            with emb_col1:
                st.markdown("**Node2Vec** (graph structure)")
                node2vec_retriever = HotelEmbeddingRetriever(st.session_state.db_manager.driver, model_type='node2vec')
                node2vec_results = node2vec_retriever.find_similar_hotels(reference_hotel, top_k=5)

                if node2vec_results:
                    n2v_data = []
                    for i, r in enumerate(node2vec_results, 1):
                        n2v_data.append({
                            'Rank': i,
                            'Hotel': r.get('hotel', r.get('name', 'Unknown')),
                            'Similarity': f"{r.get('score', 0):.4f}"
                        })
                    st.dataframe(pd.DataFrame(n2v_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No results from Node2Vec")

            with emb_col2:
                st.markdown("**FastRP** (random projection)")
                fastrp_retriever = HotelEmbeddingRetriever(st.session_state.db_manager.driver, model_type='fastrp')
                fastrp_results = fastrp_retriever.find_similar_hotels(reference_hotel, top_k=5)

                if fastrp_results:
                    frp_data = []
                    for i, r in enumerate(fastrp_results, 1):
                        frp_data.append({
                            'Rank': i,
                            'Hotel': r.get('hotel', r.get('name', 'Unknown')),
                            'Similarity': f"{r.get('score', 0):.4f}"
                        })
                    st.dataframe(pd.DataFrame(frp_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No results from FastRP")

        except Exception as e:
            st.warning(f"Could not compare embeddings: {e}")
    else:
        st.info("Mention a hotel name in your query to compare embedding results (e.g., 'Hotels similar to The Azure Tower')")

    # LLM Response
    st.subheader("J.A.R.V.I.S. Response")
    st.caption(f"Using **{model_option}** with **{embedding_option}** embeddings")
    try:
        with st.spinner(f"Generating response with {model_option}..."):
            answer_gemma, answer_mistral, answer_llama, eval_results = call_llm(
                user_question,
                baseline_context=valid_context,
                embedding_model=embedding_model
            )

        # Select answer based on model choice
        if model_option == "Gemma":
            selected_answer = answer_gemma
        elif model_option == "Mistral":
            selected_answer = answer_mistral
        else:  # LLaMA
            selected_answer = answer_llama

        st.success(selected_answer)

        # Show all model responses with metrics
        with st.expander("View All Model Responses & Metrics"):
            # Display comparison table
            st.markdown("### Model Comparison Metrics")
            metrics_data = []
            for model_key in ['gemma', 'mistral', 'llama']:
                if model_key in eval_results:
                    m = eval_results[model_key]
                    metrics_data.append({
                        'Model': m.get('model', model_key),
                        'Latency (s)': m.get('latency_sec', 'N/A'),
                        'Input Tokens': m.get('input_tokens', 'N/A'),
                        'Output Tokens': m.get('output_tokens', 'N/A'),
                        'Total Cost ($)': m.get('total_cost_usd', 'N/A'),
                        'Semantic Accuracy': m.get('semantic_accuracy', 'N/A')
                    })
            if metrics_data:
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

            st.markdown("---")

            # Individual responses
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Gemma-2-2B:**")
                if 'gemma' in eval_results:
                    st.caption(f"Latency: {eval_results['gemma'].get('latency_sec', 'N/A')}s | Cost: ${eval_results['gemma'].get('total_cost_usd', 'N/A')}")
                st.write(answer_gemma)

            with col2:
                st.markdown("**Mistral-7B:**")
                if 'mistral' in eval_results:
                    st.caption(f"Latency: {eval_results['mistral'].get('latency_sec', 'N/A')}s | Cost: ${eval_results['mistral'].get('total_cost_usd', 'N/A')}")
                st.write(answer_mistral)

            with col3:
                st.markdown("**LLaMA-3.1-8B:**")
                if 'llama' in eval_results:
                    st.caption(f"Latency: {eval_results['llama'].get('latency_sec', 'N/A')}s | Cost: ${eval_results['llama'].get('total_cost_usd', 'N/A')}")
                st.write(answer_llama)

    except Exception as e:
        st.error(f"Error calling LLM: {e}")

else:
    # Welcome message when no query
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #888;">
        <h3>Welcome, Sir.</h3>
        <p>I'm at your service. Ask me about hotels, recommendations, reviews, or comparisons.</p>
        <p>Select a quick query above or type your own question.</p>
    </div>
    """, unsafe_allow_html=True)
