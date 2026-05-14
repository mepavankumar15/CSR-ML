import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

class ModelEvaluator:
    """
    Evaluates recommender system performance and generates reports.
    """
    def __init__(self):
        pass

    def train_test_split_temporal(self, df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
        """
        Split dataset chronologically into train and test sets.
        
        Args:
            df (pd.DataFrame): Full transaction DataFrame.
            test_ratio (float): Proportion of transactions for test set.
            
        Returns:
            tuple: (train_df, test_df)
        """
        df_sorted = df.sort_values('transaction_date')
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        return train_df, test_df

    def evaluate_collaborative(self, recommender, test_df: pd.DataFrame, k: int = 5) -> dict:
        """
        Evaluate collaborative filtering using Precision@K, Recall@K, etc.
        
        Args:
            recommender: Fitted CrossSellRecommender instance.
            test_df (pd.DataFrame): Test transactions.
            k (int): Number of recommendations to evaluate.
            
        Returns:
            dict: Evaluation metrics.
        """
        # Group test products by customer
        test_purchases = test_df[test_df['is_returned'] == 0].groupby('customer_id')['product_id'].apply(set).to_dict()
        
        precisions, recalls, hits, ndcgs = [], [], [], []
        
        # Only evaluate for customers who exist in both train (recommender knowledge) and test
        valid_customers = [cid for cid in test_purchases.keys() if cid in recommender.matrix.index]
        
        # Sample a subset to speed up evaluation in interactive usage
        eval_customers = np.random.choice(valid_customers, size=min(100, len(valid_customers)), replace=False)
        
        for cid in eval_customers:
            actual_items = test_purchases[cid]
            if not actual_items:
                continue
                
            recs = recommender.recommend(cid, method='collaborative', top_n=k)
            # Recs are dicts: {'product_id': ..., 'score': ...}
            rec_items = [r['product_id'] for r in recs]
            
            # Hits
            hits_in_k = len(set(rec_items) & actual_items)
            
            # Precision & Recall
            precisions.append(hits_in_k / k)
            recalls.append(hits_in_k / len(actual_items))
            hits.append(1 if hits_in_k > 0 else 0)
            
            # NDCG@K
            dcg = 0
            idcg = 0
            for i, item in enumerate(rec_items):
                if item in actual_items:
                    dcg += 1 / np.log2(i + 2)  # +2 because i is 0-indexed
            
            for i in range(min(len(actual_items), k)):
                idcg += 1 / np.log2(i + 2)
                
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
        avg_p = np.mean(precisions) if precisions else 0
        avg_r = np.mean(recalls) if recalls else 0
        f1 = 2 * (avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        
        return {
            'Precision@K': round(avg_p, 4),
            'Recall@K': round(avg_r, 4),
            'F1@K': round(f1, 4),
            'Hit Rate': round(np.mean(hits), 4) if hits else 0,
            'NDCG@K': round(np.mean(ndcgs), 4) if ndcgs else 0
        }

    def evaluate_association_rules(self, rules_df: pd.DataFrame) -> dict:
        """
        Evaluate association rules quality.
        
        Args:
            rules_df (pd.DataFrame): DataFrame of rules from mlxtend.
            
        Returns:
            dict: Summary metrics.
        """
        if rules_df is None or rules_df.empty:
            return {"Number of Rules": 0}
            
        return {
            "Number of Rules": len(rules_df),
            "Avg Support": round(rules_df['support'].mean(), 4),
            "Avg Confidence": round(rules_df['confidence'].mean(), 4),
            "Avg Lift": round(rules_df['lift'].mean(), 4)
        }

    def plot_evaluation_results(self, metrics_dict: dict) -> go.Figure:
        """
        Plot bar chart comparing models across metrics.
        
        Args:
            metrics_dict (dict): Dictionary mapping model names to metric dicts.
                e.g. {'Collaborative': {'Precision@K': 0.1, ...}, 'Hybrid': {...}}
                
        Returns:
            go.Figure: Plotly figure.
        """
        # Prepare data for plotting
        models = list(metrics_dict.keys())
        # Assuming all models have the same metrics evaluated
        if not models:
            return go.Figure()
            
        metrics_names = list(metrics_dict[models[0]].keys())
        
        fig = go.Figure()
        
        for metric in metrics_names:
            if metric == 'Number of Rules': # Skip non-comparable metrics
                continue
                
            values = [metrics_dict[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(
                x=models,
                y=values,
                name=metric,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title="Model Comparison across Metrics",
            barmode='group',
            xaxis_title="Models",
            yaxis_title="Score",
            template='plotly_dark',
            legend_title="Metrics"
        )
        return fig

    def coverage_report(self, recommender, df: pd.DataFrame, top_k=5) -> dict:
        """
        Calculate catalog and customer coverage.
        
        Args:
            recommender: Fitted recommender instance.
            df (pd.DataFrame): Transaction data.
            top_k (int): Number of recommendations to consider per customer.
            
        Returns:
            dict: Coverage metrics.
        """
        all_products = set(df['product_id'].unique())
        customers = recommender.matrix.index.tolist()
        
        # Sample customers to keep computation reasonable
        eval_customers = np.random.choice(customers, size=min(200, len(customers)), replace=False)
        
        recommended_items = set()
        customers_with_recs = 0
        
        for cid in eval_customers:
            recs = recommender.recommend(cid, method='hybrid', top_n=top_k)
            if recs:
                customers_with_recs += 1
                recommended_items.update([r['product_id'] for r in recs])
                
        catalog_coverage = len(recommended_items) / len(all_products) if all_products else 0
        customer_coverage = customers_with_recs / len(eval_customers) if eval_customers.size else 0
        
        return {
            'Catalog Coverage': round(catalog_coverage, 4),
            'Customer Coverage': round(customer_coverage, 4)
        }
