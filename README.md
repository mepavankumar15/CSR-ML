# Cross-Sell Recommender System

## Overview
This project is an end-to-end Machine Learning web application designed to demonstrate a **Cross-Sell Recommender System**. In e-commerce settings, identifying cross-selling opportunities (predicting what additional products a user might want to buy) can drive substantial revenue. This project utilizes synthetic transaction data to simulate real-world e-commerce behaviors and provides a multi-strategy recommender system with an interactive Streamlit dashboard.

## Architecture
```text
[ Data Generator ] --> [ sample_transactions.csv ]
                                |
                                v
                      [ Preprocessing Module ]
                      - Clean Data
                      - One-Hot Encode
                      - Scale Features
                                |
                                v
               [ Cross-Sell Recommender Engine ]
               /            |          |           \
     Collaborative     Association   Content     Hybrid
       Filtering         Rules      Filtering    Ensemble
              \             |          |           /
               \            v          v          /
                \-->  [ Model Evaluator ] <------/
                                |
                                v
                   [ Streamlit Web Dashboard ]
```

## Models Used

| Model | Algorithm | When Best | Limitation |
|---|---|---|---|
| **Collaborative Filtering** | User-Based Cosine Similarity | Works great for finding what "people like you" also bought. | Cold-start problem for new users/items; sparse matrix issues. |
| **Association Rules** | Apriori Algorithm | Finding global patterns (e.g. A is often bought with B). | Doesn't personalize beyond the current basket. |
| **Content-Based Filtering** | Item Feature Cosine Similarity | Great for new items (no item cold-start); focuses on item attributes. | Over-specialization (Filter Bubble); hard to explore cross-category. |
| **Hybrid Ensemble** | Weighted Score Combination | Most robust for real-world applications; balances out the flaws of individual models. | Computationally more expensive; requires tuning weights. |

## Setup and Installation
1. **Clone the repository**
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate Synthetic Data:**
   ```bash
   python data/generate_data.py
   ```
4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Streamlit multi-page UI entry point.
- `requirements.txt`: Dependencies.
- `README.md`: Project documentation.
- `data/generate_data.py`: Synthetic dataset generator.
- `src/preprocessing.py`: Data cleaning & feature engineering.
- `src/recommender.py`: Core ML models (Collaborative, Association, Content, Hybrid).
- `src/evaluator.py`: Metrics & evaluation framework.
- `src/utils.py`: Helper functions for formatting and model saving/loading.
- `notebooks/01_EDA.ipynb`: Exploratory Data Analysis.
- `notebooks/02_ML_Pipeline.ipynb`: Full ML pipeline with explanatory comments.

## Screenshots
*(Add screenshots of the dashboard, recommendations, and EDA insights here)*
