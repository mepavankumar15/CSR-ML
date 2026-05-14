import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import os

from src.preprocessing import DataPreprocessor

class CrossSellRecommender:
    """
    Wraps FOUR recommendation strategies: Collaborative, Association Rule, Content-Based, and Hybrid.
    """
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.matrix = None
        self.customer_meta = None
        self.product_features = None
        self.rules_df = None
        self.product_details = {}
        
        # Similarities
        self.user_similarity = None
        self.item_similarity = None

    def fit(self, df: pd.DataFrame):
        """
        Train all models and store matrices.
        
        Args:
            df (pd.DataFrame): Raw transaction DataFrame.
        """
        print("Training Recommender Models...")
        
        # 1. Build Base Matrices
        self.matrix, self.customer_meta = self.preprocessor.get_customer_product_matrix(df)
        basket_sets = self.preprocessor.get_basket_data(df)
        
        # Build product details dictionary for quick lookup
        prod_info = df[['product_id', 'product_name', 'product_category', 'unit_price']].drop_duplicates('product_id')
        self.product_details = prod_info.set_index('product_id').to_dict('index')
        
        # --- MODEL 1: Collaborative Filtering (User-Based) ---
        print("Building Collaborative Filtering Model...")
        # Compute cosine similarity between customers (sparse to dense conversion handled by sklearn if needed)
        # Using a small subset or optimized sparse matrices in real prod, but matrix is small enough here.
        self.user_similarity = cosine_similarity(self.matrix)
        self.user_similarity = pd.DataFrame(self.user_similarity, index=self.matrix.index, columns=self.matrix.index)
        
        # --- MODEL 2: Association Rule Mining (Apriori) ---
        print("Mining Association Rules...")
        self.train_association_rules(basket_sets, min_support=0.01, min_confidence=0.1, min_lift=1.0)
        
        # --- MODEL 3: Content-Based Filtering ---
        print("Building Content-Based Model...")
        self._build_content_features(prod_info)
        self.item_similarity = cosine_similarity(self.product_features)
        self.item_similarity = pd.DataFrame(self.item_similarity, index=self.product_features.index, columns=self.product_features.index)
        
        print("Training Complete.")

    def _build_content_features(self, prod_info: pd.DataFrame):
        """Helper to build product feature vectors."""
        # Categorical feature: category
        category_dummies = pd.get_dummies(prod_info['product_category'])
        
        # Numerical feature: standardized price
        price_std = (prod_info['unit_price'] - prod_info['unit_price'].mean()) / prod_info['unit_price'].std()
        
        features = pd.concat([category_dummies, price_std.rename('price')], axis=1)
        features.index = prod_info['product_id']
        self.product_features = features

    def find_similar_customers(self, customer_id: str, top_n: int = 10) -> list:
        """Find similar customers using precomputed cosine similarity."""
        if customer_id not in self.user_similarity.index:
            return []
            
        sim_scores = self.user_similarity[customer_id].drop(customer_id)
        top_similar = sim_scores.nlargest(top_n)
        return list(zip(top_similar.index, top_similar.values))

    def recommend_collaborative(self, customer_id: str, top_n: int = 5) -> list:
        """Recommend products bought by similar customers."""
        if customer_id not in self.matrix.index:
            return [] # Cold start
            
        user_vector = self.matrix.loc[customer_id]
        already_bought = user_vector[user_vector > 0].index.tolist()
        
        similar_users = self.find_similar_customers(customer_id, top_n=20)
        
        rec_scores = {}
        
        for sim_user, score in similar_users:
            sim_user_vector = self.matrix.loc[sim_user]
            sim_bought = sim_user_vector[sim_user_vector > 0].index.tolist()
            
            for item in sim_bought:
                if item not in already_bought:
                    if item not in rec_scores:
                        rec_scores[item] = 0
                    rec_scores[item] += score * sim_user_vector[item]
                    
        # Sort and return
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{'product_id': p, 'score': s, 'source': 'Collaborative'} for p, s in sorted_recs]

    def train_association_rules(self, basket_df: pd.DataFrame, min_support=0.01, min_confidence=0.1, min_lift=1.0):
        """Mine frequent itemsets and generate rules."""
        frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            self.rules_df = pd.DataFrame()
            return
            
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
        # Filter by confidence
        rules = rules[rules['confidence'] >= min_confidence]
        self.rules_df = rules.sort_values('lift', ascending=False)

    def recommend_association(self, bought_products: list, top_n: int = 5) -> list:
        """Recommend consequents based on antecedent subsets."""
        if self.rules_df is None or self.rules_df.empty or not bought_products:
            return []
            
        bought_set = frozenset(bought_products)
        rec_scores = {}
        
        for _, row in self.rules_df.iterrows():
            antecedents = row['antecedents']
            if antecedents.issubset(bought_set):
                for item in row['consequents']:
                    if item not in bought_products:
                        if item not in rec_scores or row['lift'] > rec_scores[item]:
                            rec_scores[item] = row['lift']
                            
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{'product_id': p, 'score': s, 'source': 'Association'} for p, s in sorted_recs]

    def recommend_content_based(self, bought_products: list, top_n: int = 5) -> list:
        """Recommend products similar to what user has bought based on features."""
        if not bought_products or self.item_similarity is None:
            return []
            
        valid_products = [p for p in bought_products if p in self.item_similarity.index]
        if not valid_products:
            return []
            
        rec_scores = {}
        for item in self.item_similarity.index:
            if item not in valid_products:
                # Average similarity to bought products
                sims = [self.item_similarity.loc[item, bp] for bp in valid_products]
                avg_sim = np.mean(sims)
                rec_scores[item] = avg_sim
                
        sorted_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{'product_id': p, 'score': s, 'source': 'Content-Based'} for p, s in sorted_recs]

    def recommend_hybrid(self, customer_id: str, bought_products: list, 
                         weights={'collab': 0.4, 'assoc': 0.35, 'content': 0.25}, top_n: int = 5) -> list:
        """Combine recommendations using a weighted sum of normalized scores."""
        recs_collab = self.recommend_collaborative(customer_id, top_n=20)
        recs_assoc = self.recommend_association(bought_products, top_n=20)
        recs_content = self.recommend_content_based(bought_products, top_n=20)
        
        # Normalize scores to 0-1 within each source
        def normalize(recs):
            if not recs: return {}
            max_val = max(r['score'] for r in recs)
            min_val = min(r['score'] for r in recs)
            diff = max_val - min_val if max_val != min_val else 1
            return {r['product_id']: (r['score'] - min_val) / diff for r in recs}
            
        norm_collab = normalize(recs_collab)
        norm_assoc = normalize(recs_assoc)
        norm_content = normalize(recs_content)
        
        all_candidates = set(list(norm_collab.keys()) + list(norm_assoc.keys()) + list(norm_content.keys()))
        
        final_scores = []
        for item in all_candidates:
            score = (norm_collab.get(item, 0) * weights['collab'] +
                     norm_assoc.get(item, 0) * weights['assoc'] +
                     norm_content.get(item, 0) * weights['content'])
            
            # Determine dominant source for attribution
            sources = {'Collaborative': norm_collab.get(item, 0), 
                       'Association': norm_assoc.get(item, 0), 
                       'Content-Based': norm_content.get(item, 0)}
            dominant_source = max(sources, key=sources.get) if any(sources.values()) else 'Hybrid'
            
            final_scores.append({'product_id': item, 'score': score, 'source': dominant_source})
            
        sorted_recs = sorted(final_scores, key=lambda x: x['score'], reverse=True)[:top_n]
        return sorted_recs

    def recommend(self, customer_id: str, method='hybrid', top_n=5) -> list:
        """Unified entry point for recommendations."""
        bought_products = []
        if self.matrix is not None and customer_id in self.matrix.index:
            user_vector = self.matrix.loc[customer_id]
            bought_products = user_vector[user_vector > 0].index.tolist()
            
        # Cold start fallback
        if not bought_products:
            # Popularity based fallback
            if self.matrix is not None:
                popular = self.matrix.sum().nlargest(top_n).index.tolist()
                return [{'product_id': p, 'score': 1.0, 'source': 'Popularity (Cold Start)'} for p in popular]
            return []

        if method == 'collaborative':
            return self.recommend_collaborative(customer_id, top_n)
        elif method == 'association':
            return self.recommend_association(bought_products, top_n)
        elif method == 'content':
            return self.recommend_content_based(bought_products, top_n)
        else: # hybrid
            return self.recommend_hybrid(customer_id, bought_products, top_n=top_n)

    def get_product_details(self, product_ids: list) -> list:
        """Return product details for display."""
        details = []
        for pid in product_ids:
            if pid in self.product_details:
                d = self.product_details[pid].copy()
                d['product_id'] = pid
                details.append(d)
        return details

    def save_models(self, path: str):
        """Save instance using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Models saved to {path}")

    @staticmethod
    def load_models(path: str):
        """Load instance using joblib."""
        if os.path.exists(path):
            return joblib.load(path)
        return None
