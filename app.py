import streamlit as st
from entity_extractor import extract_entities, get_cypher_params
from chatbot import process_query_for_baseline, call_llm
from db_manager import DBManager
from intent_classifier import classify_intent
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd

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
                        label = f"Review (★{props.get('score_overall', 'N/A')})"
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


st.title("Hotel Recommender System")

# Model selection row
col_llm, col_embed = st.columns(2)

with col_llm:
    model_option = st.selectbox(
        "Select LLM Model:",
        options=["Gemma", "Mistral", "LLaMA"],
        index=0
    )

with col_embed:
    embedding_option = st.selectbox(
        "Select Node Embedding Model:",
        options=["Node2Vec (128-dim)", "FastRP (128-dim)"],
        index=0,
        help="Node2Vec: Random walk-based graph embeddings. Captures neighborhood structure through simulated walks.\n\nFastRP: Fast Random Projection embeddings. Faster computation using random projections."
    )
    # Map display name to model type
    embedding_model = 'node2vec' if 'Node2Vec' in embedding_option else 'fastrp'

# Query input
user_question = st.text_input("Ask me anything about hotels:")

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
            with st.expander(f"Context {idx}"):
                # Show cypher query
                query_text = ctx.get('query') if isinstance(ctx, dict) else None
                if query_text:
                    st.code(query_text, language='cypher')

                # Render rows (aggregations) or graph
                if isinstance(ctx, dict) and 'rows' in ctx and ctx['rows']:
                    st.dataframe(pd.DataFrame(ctx['rows']))
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

    # LLM Response
    st.subheader("AI Response")
    st.caption(f"Using **{embedding_option}** embeddings")
    try:
        with st.spinner(f"Generating response with {model_option} + {embedding_option}..."):
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
        with st.expander("View All Model Responses"):
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
                    st.caption(f"⏱️ {eval_results['gemma'].get('latency_sec', 'N/A')}s | 💰 ${eval_results['gemma'].get('total_cost_usd', 'N/A')}")
                st.write(answer_gemma)

            with col2:
                st.markdown("**Mistral-7B:**")
                if 'mistral' in eval_results:
                    st.caption(f"⏱️ {eval_results['mistral'].get('latency_sec', 'N/A')}s | 💰 ${eval_results['mistral'].get('total_cost_usd', 'N/A')}")
                st.write(answer_mistral)

            with col3:
                st.markdown("**LLaMA-3.1-8B:**")
                if 'llama' in eval_results:
                    st.caption(f"⏱️ {eval_results['llama'].get('latency_sec', 'N/A')}s | 💰 ${eval_results['llama'].get('total_cost_usd', 'N/A')}")
                st.write(answer_llama)

    except Exception as e:
        st.error(f"Error calling LLM: {e}")

