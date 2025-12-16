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

# Query input
user_question = st.text_input("Ask me anything about hotels:")

if user_question:
    # LLM Response
    st.subheader("AI Response")
    # call_llm currently does not return a final answer; call it to get structured info if implemented
    try:
        llm_payload = call_llm(user_question)
        if llm_payload and isinstance(llm_payload, dict) and 'answer' in llm_payload:
            st.success(llm_payload['answer'])
        else:
            st.info("LLM response not available. call_llm() may need to return {'answer': ...}")
    except Exception as e:
        st.error(f"Error calling LLM: {e}")

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
