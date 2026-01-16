"""
Glance Fashion Search - Interactive Streamlit Demo
Triple-stream fashion retrieval system with dynamic query-time weighting
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from retriever.retriever import TripleStreamRetriever
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="Glance Fashion Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - v2
st.markdown(
    """
<style>
    .main-header {
        font-size: 5.5rem !important;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem !important;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.5rem !important;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize retriever
if 'retriever' not in st.session_state:
    with st.spinner('🔄 Loading CLIP model and ChromaDB collections...'):
        st.session_state.retriever = TripleStreamRetriever()
        st.session_state.initialized = True

# Header
st.markdown('<p class="main-header">👗 Glance Fashion Search</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Triple-Stream Multimodal Fashion Retrieval</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    preset_options = {
        "Compositional": "compositional",
        "Attribute Specific": "attribute_specific",
        "Contextual/Place": "contextual_place",
        "Complex Semantic": "complex_semantic",
        "Style Inference": "style_inference",
        "Custom Weights": "custom"
    }
    
    preset_choice = st.selectbox(
        "Weight Preset",
        list(preset_options.keys()),
        help="Pre-configured weights for different query types"
    )
    
    preset = preset_options[preset_choice]
    
    if preset == "custom":
        st.subheader("Custom Weights")
        alpha = st.slider("α (Grounded)", 0.0, 1.0, 0.33, 0.01)
        beta = st.slider("β (Vibe)", 0.0, 1.0, 0.33, 0.01)
        gamma = st.slider("γ (Visual)", 0.0, 1.0, 0.34, 0.01)
        preset = None
    else:
        preset_weights = st.session_state.retriever.weight_presets.get(preset, {})
        if preset_weights:
            st.info(f"""
            **Weights:**
            - α (Grounded): {preset_weights['alpha']}
            - β (Vibe): {preset_weights['beta']}
            - γ (Visual): {preset_weights['gamma']}
            """)
        alpha, beta, gamma = 0.33, 0.33, 0.34
    
    st.subheader("Options")
    top_k = st.slider("Results", 5, 20, 10)
    expand_query = st.checkbox("Query Expansion", value=True)
    show_metadata = st.checkbox("Show Metadata", value=True)
    
    st.subheader("Filters")
    use_filters = st.checkbox("Filter by Garments")
    if use_filters:
        min_garments = st.number_input("Min", 0, 20, 0)
        max_garments = st.number_input("Max", 0, 20, 20)
        filters = {'min_garments': min_garments, 'max_garments': max_garments}
    else:
        filters = None
    
    st.markdown("---")
    st.caption("✅ 3 Collections | 45,623 vectors each")
    st.caption("✅ clip-ViT-B-32")
    st.caption("⚡ Avg latency: ~45-50ms")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search", "📝 Examples", "ℹ️ About"])

with tab1:
    query = st.text_input(
        "Enter your fashion search query:",
        placeholder="e.g., A person in a bright yellow raincoat"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        with st.spinner('🔍 Searching 45,623 images...'):
            start_time = time.time()
            
            # Store original query for comparison
            original_query = query
            
            if preset:
                results = st.session_state.retriever.dynamic_search(
                    query, preset=preset, top_k=top_k, expand=expand_query, filters=filters
                )
            else:
                results = st.session_state.retriever.dynamic_search(
                    query, alpha=alpha, beta=beta, gamma=gamma, 
                    top_k=top_k, expand=expand_query, filters=filters
                )
            
            results = st.session_state.retriever.rerank_by_color(results, query)
            latency_ms = (time.time() - start_time) * 1000
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Results Found", len(results))
        with col2:
            st.metric("Total Latency", f"{latency_ms:.2f} ms")
        with col3:
            avg_score = sum(r[1] for r in results) / len(results) if results else 0
            st.metric("Avg Score", f"{avg_score:.4f}")
        with col4:
            st.metric("Collections", "3 streams")
        
        # Query expansion info
        if expand_query:
            expanded = st.session_state.retriever.expand_query(original_query)
            if expanded != original_query:
                st.info(f"🔄 **Query Expansion:** `{original_query}` → `{expanded}`")
        
        # Stream score statistics
        if results:
            avg_grounded = sum(r[2]['grounded'] for r in results) / len(results)
            avg_vibe = sum(r[2]['vibe'] for r in results) / len(results)
            avg_visual = sum(r[2]['visual'] for r in results) / len(results)
            
            st.markdown("**Stream Performance:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Grounded", f"{avg_grounded:.3f}")
            with col2:
                st.metric("Avg Vibe", f"{avg_vibe:.3f}")
            with col3:
                st.metric("Avg Visual", f"{avg_visual:.3f}")
        
        st.markdown("---")
        
        image_ids = [r[0] for r in results]
        metadata_batch = st.session_state.retriever.get_batch_metadata(image_ids)
        
        for rank, (img_id, score, scores) in enumerate(results, 1):
            metadata = metadata_batch.get(img_id, {})
            image_path = metadata.get('image_path', '')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if os.path.exists(image_path):
                    try:
                        st.image(Image.open(image_path), use_container_width=True)
                    except:
                        st.error("Image error")
                else:
                    st.warning("Image unavailable")
            
            with col2:
                st.markdown(f"### Rank #{rank}")
                st.markdown(f"**Score: {score:.4f}**")
                
                col_g, col_v, col_i = st.columns(3)
                with col_g:
                    st.metric("Grounded", f"{scores['grounded']:.3f}")
                with col_v:
                    st.metric("Vibe", f"{scores['vibe']:.3f}")
                with col_i:
                    st.metric("Visual", f"{scores['visual']:.3f}")
                
                if show_metadata:
                    if 'num_garments' in metadata:
                        st.caption(f"👔 Garments: {metadata['num_garments']}")
                    if 'grounded_text' in metadata:
                        with st.expander("Attributes"):
                            st.text(metadata['grounded_text'][:500])
                    if 'vibe_text' in metadata:
                        with st.expander("Context"):
                            st.text(metadata['vibe_text'][:500])
                    st.caption(f"ID: {img_id}")
            
            st.markdown("---")

with tab2:
    st.header("📝 Example Queries")
    
    examples = [
        ("A person in a bright yellow raincoat", "attribute_specific"),
        ("Professional business attire inside a modern office", "contextual_place"),
        ("Someone wearing a blue shirt sitting on a park bench", "complex_semantic"),
        ("Casual weekend outfit for a city walk", "style_inference"),
        ("A red tie and a white shirt in a formal setting", "compositional")
    ]
    
    for i, (ex_query, ex_preset) in enumerate(examples, 1):
        with st.expander(f"**{i}. {ex_query}**"):
            st.markdown(f"**Preset:** `{ex_preset}`")
            if st.button("Try", key=f"ex_{i}"):
                st.session_state.example_query = ex_query
                st.rerun()

with tab3:
    st.header("ℹ️ About")
    st.markdown("""
    ### Triple-Stream Architecture
    
    The system understands:
    - **What** - Clothing attributes, colors, garments (V_fact)
    - **Where** - Context, setting, environment (V_vibe)
    - **Vibe** - Style, occasion, mood (V_vibe)
    
    ### Scoring Formula
    ```
    Final Score = α × S_grounded + β × S_vibe + γ × S_visual
    ```
    
    ### Dataset
    - **Images**: 45,623 (Fashionpedia)
    - **Vectors**: 136,869 total (3 streams × 45,623)
    - **Storage**: ChromaDB persistent database
    
    ### Performance
    - First query: ~200-250ms (warmup)
    - Average: ~45-50ms
    - Scalable to millions of images
    
    ### Technology
    - CLIP (text/image embeddings)
    - BLIP-2 (scene captions)
    - ChromaDB (vector database)
    - Fashionpedia (fashion attributes)
    """)

if 'example_query' in st.session_state:
    query = st.session_state.example_query
    del st.session_state.example_query
