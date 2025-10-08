#  Synthetic Customer Profiles for Product Recommendation
### 6-Week Internship Project | Global Next Consulting India Pvt. Ltd.
**Mentor:** Ms. Abhipsa Guha  
**Team Members:** Aditi • Disha • Aarayan • Himanshu • Lakshya  



##  Project Overview
This project focuses on generating synthetic customer data using **CTGAN (Generative AI)** and evaluating its performance in **product recommendation models** such as Random Forest, XGBoost, and LightGBM.

Synthetic data helps ensure **data privacy** while maintaining the statistical characteristics of real data.



##  Workflow
1. Data Preprocessing & Cleaning  
2. CTGAN-based Synthetic Data Generation  
3. Exploratory Data Analysis (EDA)  
4. Model Training & Evaluation  
5. Result Visualization & Insights  



##  Tools & Technologies
| Category | Tools |
|-----------|--------|
| Language | Python 3.10 |
| Libraries | Pandas, NumPy, Plotly, Scikit-learn, SDV, XGBoost, LightGBM |
| Environment | Google Colab |
| Visualization | Plotly Express, Seaborn |
| Models | Random Forest, XGBoost, LightGBM, CTGAN |


##  Dataset
- **Source:** shopping_trends_updated.csv (custom dataset)
- **Synthetic Generation:** CTGAN trained for 500 epochs
- **Merged Data Size:** ~8,000 records



##  Results Summary
| Model | Data Type | Accuracy |Top 3 Accuracy|
|--------|------------|-----------|------------|
| Random Forest | Augmented data  | 0.4387% |0.9004 |
| XGBoost | Augmented data  |0.4487	% |0.9050|
| LightGBM | Augmented data  | 0.4579% |0.9059|


