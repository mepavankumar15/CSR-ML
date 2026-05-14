import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time

from src.recommender import CrossSellRecommender
from src.evaluator import ModelEvaluator
from src.utils import format_currency, get_category_icon, truncate_name, color_by_score, load_or_train_models

# --- CONFIGURATION ---
st.set_page_config(page_title="Cross-Sell Recommender", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark Theme Optimization */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Mono', monospace;
        color: #00D4FF;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background-color: #1E2329;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2B3139;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00D4FF;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #A0AEC0;
    }
    
    /* Product Card styling */
    .product-card {
        background-color: #1E2329;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #00D4FF;
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-2px);
    }
    
    .score-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    
    /* Badges */
    .badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .badge-premium { background: linear-gradient(90deg, #FFD700, #FFA500); color: black; }
    .badge-standard { background: #4A5568; color: white; }
    .badge-budget { background: #E53E3E; color: white; }
    
    .source-collab { background: #3182CE; color: white; }
    .source-assoc { background: #805AD5; color: white; }
    .source-content { background: #DD6B20; color: white; }
    .source-hybrid { background: #38A169; color: white; }
    .source-popularity { background: #718096; color: white; }

</style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING ---
@st.cache_data
def load_data():
    file_path = "data/sample_transactions.csv"
    if not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

@st.cache_resource
def get_recommender():
    data_path = "data/sample_transactions.csv"
    model_path = "models/recommender.pkl"
    if not os.path.exists(data_path):
        return None
    return load_or_train_models(data_path, model_path)

df = load_data()
recommender = get_recommender()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛒 CSR System")
page = st.sidebar.radio("Navigation", ["Dashboard", "Get Recommendations", "Model Performance", "EDA Insights", "Settings and Data"])

# Fallback if no data
if df.empty and page != "Settings and Data":
    st.warning("⚠️ No data found. Please run the data generator script or go to 'Settings and Data' to generate data.")
    st.stop()
if recommender is None and page != "Settings and Data":
    st.warning("⚠️ Model not trained. Please go to 'Settings and Data' to retrain.")
    st.stop()

# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if page == "Dashboard":
    st.title("📊 Executive Dashboard")
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Customers</div><div class="metric-value">{df["customer_id"].nunique():,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Products</div><div class="metric-value">{df["product_id"].nunique():,}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Transactions</div><div class="metric-value">{df["transaction_id"].nunique():,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Revenue</div><div class="metric-value">{format_currency(df["total_price"].sum())}</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Top 10 Products
        top_prods = df.groupby('product_name')['total_price'].sum().nlargest(10).reset_index()
        fig1 = px.bar(top_prods, x='total_price', y='product_name', orientation='h', template='plotly_dark', 
                      title="Top 10 Best-Selling Products by Revenue", color='total_price', color_continuous_scale='Viridis')
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        # Revenue by Category
        rev_cat = df.groupby('product_category')['total_price'].sum().reset_index()
        fig2 = px.pie(rev_cat, values='total_price', names='product_category', hole=0.4, template='plotly_dark',
                      title="Revenue by Category", color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig2, use_container_width=True)
        
    # Monthly Sales Trend
    monthly = df.set_index('transaction_date').resample('ME')['total_price'].sum().reset_index()
    fig3 = px.line(monthly, x='transaction_date', y='total_price', markers=True, template='plotly_dark',
                   title="Monthly Sales Trend", line_shape='spline')
    fig3.update_traces(line_color='#00D4FF', line_width=3)
    st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# PAGE 2: GET RECOMMENDATIONS
# ==========================================
elif page == "Get Recommendations":
    st.title("🎯 Personalized Recommendations")
    
    # Sidebar Controls
    st.sidebar.subheader("Recommendation Settings")
    customers = sorted(df['customer_id'].unique())
    selected_customer = st.sidebar.selectbox("Select Customer ID", customers)
    
    method_map = {
        "Hybrid Ensemble": "hybrid",
        "Collaborative Filtering": "collaborative",
        "Association Rules": "association",
        "Content-Based": "content"
    }
    selected_method_name = st.sidebar.radio("Strategy", list(method_map.keys()))
    selected_method = method_map[selected_method_name]
    
    num_recs = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
    conf_threshold = st.sidebar.slider("Confidence Threshold (Min Score)", 0.0, 1.0, 0.0, step=0.05)
    
    c_left, c_right = st.columns([1, 2])
    
    # Customer Profile
    with c_left:
        st.subheader("👤 Customer Profile")
        c_data = df[df['customer_id'] == selected_customer].copy()
        
        if len(c_data) > 0:
            info = c_data.iloc[0]
            seg = info['customer_segment']
            badge_class = f"badge-{seg.lower()}"
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: left;">
                <h3>{selected_customer}</h3>
                <p><b>Segment:</b> <span class="badge {badge_class}">{seg}</span></p>
                <p><b>Age / Gender:</b> {info['customer_age']} / {info['customer_gender']}</p>
                <p><b>City:</b> {info['city']}</p>
                <hr style="border-color: #2B3139;">
                <p><b>Total Spend:</b> {format_currency(c_data['total_price'].sum())}</p>
                <p><b>Transactions:</b> {c_data['transaction_id'].nunique()}</p>
                <p><b>Fav Category:</b> {c_data['product_category'].mode()[0]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Radar Chart
            cat_spend = c_data.groupby('product_category')['total_price'].sum().reset_index()
            fig_radar = px.line_polar(cat_spend, r='total_price', theta='product_category', line_close=True,
                                      template='plotly_dark', title="Spending Profile")
            fig_radar.update_traces(fill='toself', fillcolor='rgba(0, 212, 255, 0.3)', line_color='#00D4FF')
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.warning("Customer data not found.")
            
    # Recommendations & History
    with c_right:
        with st.expander("🛍️ Recent Purchase History (Last 10)", expanded=False):
            hist = c_data.sort_values('transaction_date', ascending=False).head(10)
            st.dataframe(hist[['transaction_date', 'product_name', 'product_category', 'quantity', 'total_price']], use_container_width=True)
            
        st.subheader("✨ Recommended Products")
        
        with st.spinner("Generating recommendations..."):
            # Get recs
            recs = recommender.recommend(selected_customer, method=selected_method, top_n=num_recs)
            
            # Filter by threshold (normalize first)
            if recs:
                max_s = max(r['score'] for r in recs)
                min_s = min(r['score'] for r in recs)
                diff = max_s - min_s if max_s != min_s else 1
                
                filtered_recs = []
                for r in recs:
                    norm_score = (r['score'] - min_s) / diff if 'Popularity' not in r['source'] else 1.0
                    if norm_score >= conf_threshold:
                        r['norm_score'] = norm_score
                        filtered_recs.append(r)
            else:
                filtered_recs = []
                
            if not filtered_recs:
                st.info("No recommendations found above the threshold. Try lowering the confidence or changing the strategy.")
            else:
                for idx, r in enumerate(filtered_recs):
                    details = recommender.get_product_details([r['product_id']])[0]
                    score = r['norm_score']
                    color = color_by_score(score)
                    
                    source_class = ""
                    if 'Collab' in r['source']: source_class = "source-collab"
                    elif 'Assoc' in r['source']: source_class = "source-assoc"
                    elif 'Content' in r['source']: source_class = "source-content"
                    elif 'Hybrid' in r['source']: source_class = "source-hybrid"
                    else: source_class = "source-popularity"
                    
                    reasoning = "Customers like you also bought this." if 'Collab' in r['source'] else \
                                "Frequently bought together with your items." if 'Assoc' in r['source'] else \
                                "Similar to products you've purchased." if 'Content' in r['source'] else \
                                "Top pick for you."
                                
                    icon = get_category_icon(details['product_category'])
                    
                    # Layout card
                    st.markdown(f"""
                    <div class="product-card">
                        <div style="display: flex; justify-content: space-between;">
                            <h4 style="margin: 0; color: white;">{icon} {details['product_name']}</h4>
                            <span style="font-weight: bold; color: #00D4FF;">{format_currency(details['unit_price'])}</span>
                        </div>
                        <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #A0AEC0;">{details['product_category']} | <span class="badge {source_class}">{r['source']}</span></p>
                        <p style="margin: 5px 0 0 0; font-size: 0.85em; font-style: italic;">{reasoning}</p>
                        <div style="margin-top: 10px;">
                            <span style="font-size: 0.8em; color: #A0AEC0;">Match Confidence: {int(score*100)}%</span>
                            <div class="score-bar" style="width: {score*100}%; background-color: {color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Add {details['product_id']} to Cart", key=f"btn_{idx}"):
                        st.toast(f"Added {details['product_name']} to cart!", icon="✅")

# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.title("🔬 Model Evaluation")
    
    st.info("Evaluation uses a temporal split (80% train, 20% test). Calculating metrics dynamically for a subset of users may take a moment.")
    
    if st.button("Run Full Evaluation Suite"):
        with st.spinner("Evaluating models..."):
            evaluator = ModelEvaluator()
            train_df, test_df = evaluator.train_test_split_temporal(df)
            
            st.success("Evaluation complete!")
            
            tab1, tab2, tab3 = st.tabs(["Collaborative Filtering", "Association Rules", "Coverage"])
            
            with tab1:
                st.subheader("Collaborative Metrics (Sample)")
                collab_metrics = evaluator.evaluate_collaborative(recommender, test_df)
                c_cols = st.columns(len(collab_metrics))
                for i, (k, v) in enumerate(collab_metrics.items()):
                    c_cols[i].metric(k, v)
                    
            with tab2:
                st.subheader("Association Rules Mining")
                assoc_metrics = evaluator.evaluate_association_rules(recommender.rules_df)
                a_cols = st.columns(len(assoc_metrics))
                for i, (k, v) in enumerate(assoc_metrics.items()):
                    a_cols[i].metric(k, v)
                    
                if recommender.rules_df is not None and not recommender.rules_df.empty:
                    st.write("Top Rules by Lift")
                    top_rules = recommender.rules_df.head(10).copy()
                    top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                    
            with tab3:
                st.subheader("Catalog & Customer Coverage")
                cov_metrics = evaluator.coverage_report(recommender, df)
                c_cols = st.columns(2)
                c_cols[0].metric("Catalog Coverage", f"{cov_metrics['Catalog Coverage']*100:.2f}%")
                c_cols[1].metric("Customer Coverage", f"{cov_metrics['Customer Coverage']*100:.2f}%")

# ==========================================
# PAGE 4: EDA INSIGHTS
# ==========================================
elif page == "EDA Insights":
    st.title("📈 Exploratory Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Customer Behavior", "Product Analysis", "Transaction Patterns"])
    
    with tab1:
        st.subheader("Customer Behavior")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df.groupby('customer_id')['transaction_id'].nunique(), 
                               title="Transactions per Customer", template='plotly_dark', color_discrete_sequence=['#00D4FF'])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            seg_dist = df.drop_duplicates('customer_id')['customer_segment'].value_counts().reset_index()
            fig = px.pie(seg_dist, values='count', names='customer_segment', title="Customer Segments", template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.subheader("Product Analysis")
        fig = px.box(df, x='product_category', y='unit_price', color='product_category', 
                     title="Price Distribution by Category", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.subheader("Transaction Patterns")
        df['day_name'] = df['transaction_date'].dt.day_name()
        df['hour'] = np.random.randint(8, 22, size=len(df)) # Simulated hour
        heatmap_data = df.groupby(['day_name', 'hour']).size().unstack(fill_value=0)
        fig = px.imshow(heatmap_data, template='plotly_dark', title="Simulated Transaction Frequency: Day vs Hour", color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 5: SETTINGS AND DATA
# ==========================================
elif page == "Settings and Data":
    st.title("⚙️ Settings & System Management")
    
    st.subheader("1. Generate / Reset Data")
    st.write("Generate a fresh batch of 50,000 synthetic transactions.")
    if st.button("Generate Synthetic Data"):
        with st.spinner("Generating dataset..."):
            os.system("python data/generate_data.py")
            st.cache_data.clear()
            st.success("Data generated successfully!")
            st.rerun()
            
    st.markdown("---")
    
    st.subheader("2. Model Retraining")
    st.write("Retrain the recommendation models on the latest dataset.")
    if st.button("Retrain Models"):
        with st.spinner("Training models... This might take a minute."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading data...")
            time.sleep(0.5)
            progress_bar.progress(20)
            
            try:
                load_or_train_models("data/sample_transactions.csv", "models/recommender.pkl", force_retrain=True)
                progress_bar.progress(100)
                status_text.text("Done!")
                st.cache_resource.clear()
                st.success("Models retrained successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {e}")
                
    st.markdown("---")
    
    st.subheader("3. Export Data")
    if not df.empty:
        csv = df.head(1000).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data Sample (CSV)",
            data=csv,
            file_name='sample_transactions.csv',
            mime='text/csv',
        )
