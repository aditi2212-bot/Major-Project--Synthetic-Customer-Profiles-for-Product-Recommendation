import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import plotly.express as px

# LOAD MODELS AND DATA
lgb_cat_model = joblib.load("lgb_category_model.pkl")
lgb_item_models = joblib.load("lgb_item_models.pkl")
encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# LOAD BOTH REAL + SYNTHETIC DATASETS
data = pd.read_csv("shopping_trends_updated.csv")

# PAGE CONFIG
st.set_page_config(page_title="AI-Powered Product Recommendation Dashboard", page_icon="☄️", layout="wide")

# Style the Streamlit
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #03045e;
    font-family: 'Helvetica', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #caf0f8;
    border-radius: 10px;
    color: #03045e;
}
h1, h2, h3, h4, h5, h6 {
    color: #023e8a !important;
    font-weight: 600;
}
p, label {
    color: #03045e !important;
    font-size: 16px;
}
div.stButton > button {
    background-color: #0077b6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 20px;
    font-weight: 500;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #00b4d8;
    color: #03045e;
}
hr {
    border: none;
    border-top: 2px solid #00b4d8;
    margin: 25px 0;
}
.block-container {
    border-radius: 12px;
    padding: 20px;
}
.js-plotly-plot .gtitle {
    text-anchor: start !important;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<h2 style="text-align:center; color:#023e8a;">AI-Powered Product Recommendation Dashboard</h2>
<p style="text-align:center; color:#0077b6;">
Synthetic customer profiling for accurate, data-driven shopping recommendations.
</p>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("Enter Your Details")
age = st.sidebar.slider("Age", 15, 70, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"])
prev_purchases = st.sidebar.slider("Previous Purchases", 0, 50, 3)
location = st.sidebar.selectbox("Location", sorted(data["Location"].unique()))

if st.sidebar.button("Get Recommendations"):
    # PREPARE USER INPUT
    user_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Season": [season],
        "Previous Purchases": [prev_purchases],
        "Location": [location],
        "Review Rating": [4.0],
        "Purchase Amount (USD)": [50],
        "Size": ["M"],
        "Color": ["Blue"],
        "Subscription Status": ["Yes"],
        "Shipping Type": ["Standard"],
        "Discount Applied": ["No"],
        "Promo Code Used": ["No"],
        "Payment Method": ["Credit Card"],
        "Frequency of Purchases": ["Monthly"]
    })

    # ENCODE + SCALE
    for col in encoders:
        if col in user_df.columns:
            user_df[col] = encoders[col].transform(user_df[col])

    numeric_cols = ['Age', 'Review Rating', 'Previous Purchases', 'Purchase Amount (USD)']
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

    # CATEGORY PREDICTION
    cat_probs = lgb_cat_model.predict_proba(user_df)[0]
    top3_cat_idx = np.argsort(cat_probs)[-3:][::-1]
    top3_categories = encoders['Category'].inverse_transform(top3_cat_idx)

    st.markdown("<h3 style='text-align:center; color:#023e8a;'>Top Predicted Product Categories</h3>", unsafe_allow_html=True)
    st.caption("These categories are ranked based on model confidence for your profile.")
    for i, cat in enumerate(top3_categories, 1):
        st.markdown(f"<p style='color:#03045e;'><strong>{i}. {cat}</strong> — Confidence Score: <code>{cat_probs[top3_cat_idx[i-1]]:.2f}</code></p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # GENDER FILTER DICTIONARY
    allowed_items_per_gender = {
        'Male': ['Shirt','Pants','Shoes','Jacket','Sweater','Hoodie','Shorts','Socks',
                 'Belt','Hat','Gloves','Scarf','Backpack','Sneakers','Coat','Boots',
                 'Sunglasses','Jeans','Watch','T-shirt'],
        'Female': ['Blouse','Dress','Skirt','Heels','Sandals','Shirt','Pants','Shoes',
                   'Jacket','Sweater','Hoodie','Shorts','Socks','Belt','Hat','Gloves',
                   'Scarf','Backpack','Sneakers','Coat','Boots','Handbag','Jewelry',
                   'Sunglasses','Watch','T-shirt']
    }

    # ITEM PREDICTION (GENDER FILTERED)
    st.markdown("<h3 style='text-align:center; color:#023e8a;'>Recommended Items Within Each Category</h3>", unsafe_allow_html=True)
    st.caption("Based on your predicted categories, here are the top items you’re most likely to purchase (gender-filtered).")

    for cat in top3_categories:
        item_model = lgb_item_models[encoders['Category'].transform([cat])[0]]
        item_probs = item_model.predict_proba(user_df)[0]
        top_idx = np.argsort(item_probs)[::-1]
        all_items = encoders['Item Purchased'].inverse_transform(top_idx)

        # Apply gender filter
        allowed_items = [item for item in all_items if item in allowed_items_per_gender[gender]]
        top3_items = allowed_items[:3] if len(allowed_items) >= 3 else allowed_items

        if top3_items:
            st.markdown(f"<p style='color:#03045e;'><strong>{cat} — </strong> {', '.join(top3_items)}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:#03045e;'><strong>{cat} — </strong>No gender-appropriate items available.</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Average Spending
    avg_prices = (
        data[data['Category'].isin(top3_categories)]
        .groupby('Category')['Purchase Amount (USD)']
        .mean()
        .reset_index()
    )
    fig1 = px.bar(
        avg_prices, x='Category', y='Purchase Amount (USD)',
        color='Category', text_auto=".2f",
        color_discrete_sequence=["#0077b6", "#0096c7", "#48cae4"],
        title="Average Spending per Category"
    )
    fig1.update_layout(
        title_x=0.0,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#03045e",
        yaxis_title="Average Purchase Amount (USD)"
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("This chart shows how much customers typically spend in each of your top recommended categories.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Seasonal Popularity
    seasonal_pop = (
        data[data['Category'].isin(top3_categories)]
        .groupby(['Season', 'Category'])
        .size()
        .reset_index(name='Count')
    )
    fig2 = px.bar(
        seasonal_pop, x='Season', y='Count', color='Category',
        barmode='group', title="Seasonal Popularity Trends",
        color_discrete_sequence=["#0077b6", "#0096c7", "#48cae4"]
    )
    fig2.update_layout(
        title_x=0.0,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#03045e",
        yaxis_title="Number of Purchases"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("This chart highlights how product demand changes across different seasons for your top categories.")

    # Top 5 Location By Average Purchase
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#023e8a;'>Top 5 Locations by Average Purchase</h3>", unsafe_allow_html=True)

    top_locations = (
        data.groupby('Location')['Purchase Amount (USD)']
        .mean().nlargest(5).reset_index()
    )

    fig3 = px.bar(
        top_locations,
        x='Purchase Amount (USD)',
        y='Location',
        orientation='h',
        color='Location',
        color_discrete_sequence=["#0077b6", "#0096c7", "#00b4d8", "#48cae4", "#023e8a"],
        title="Top 5 Locations by Average Purchase"
    )

    fig3.update_layout(
        title_x=0.0,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font_color="#03045e",
        xaxis_title="Average Purchase Amount (USD)",
        yaxis_title=None,
        showlegend=False
    )

    st.plotly_chart(fig3, use_container_width=True)
    st.caption("This chart shows the top 5 locations where customers spend the most on average.")

else:
    st.markdown("<p style='text-align:center; color:#0077b6;'>Fill in your details on the left and click <strong>Get Recommendations</strong> to view results.</p>", unsafe_allow_html=True)
