import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

class DataPreprocessor:
    """
    Handles data loading, cleaning, and feature engineering for the Cross-Sell Recommender System.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.product_le = LabelEncoder()
        self.product_name_le = LabelEncoder()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV, parse dates, and validate columns.
        
        Args:
            filepath (str): Path to the CSV file.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = pd.read_csv(filepath)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        
        # Basic validation
        required_cols = ['customer_id', 'transaction_id', 'transaction_date', 'product_id', 'quantity', 'total_price']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Drop entirely missing rows if any
        df = df.dropna(how='all')
        return df
        
    def get_customer_product_matrix(self, df: pd.DataFrame) -> tuple:
        """
        Pivot table: customers as rows, products as columns.
        Values = total quantity purchased (fill NaN with 0).
        
        Args:
            df (pd.DataFrame): Transaction DataFrame.
            
        Returns:
            tuple: (pivot_matrix as DataFrame, customer metadata DataFrame)
        """
        # Exclude returned items for recommendation purposes
        df_valid = df[df['is_returned'] == 0]
        
        matrix = pd.pivot_table(
            df_valid, 
            values='quantity', 
            index='customer_id', 
            columns='product_id', 
            aggfunc='sum', 
            fill_value=0
        )
        
        # Extract customer metadata (using the first occurrence of their metadata)
        meta_cols = ['customer_id', 'customer_age', 'customer_gender', 'customer_segment', 'city']
        customer_meta = df[meta_cols].drop_duplicates(subset=['customer_id']).set_index('customer_id')
        
        return matrix, customer_meta
        
    def get_basket_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group transactions by transaction_id to create one-hot encoded baskets.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame.
            
        Returns:
            pd.DataFrame: Baskets of shape (n_transactions, n_products).
        """
        df_valid = df[df['is_returned'] == 0]
        basket = (df_valid.groupby(['transaction_id', 'product_id'])['quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('transaction_id'))
        
        # Convert positive values to 1, otherwise 0
        def encode_units(x):
            return x >= 1
        
        basket_sets = basket.map(encode_units)
        return basket_sets

    def get_customer_features(self, df: pd.DataFrame) -> tuple:
        """
        Aggregate per-customer features and scale/encode them.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame.
            
        Returns:
            tuple: (Feature matrix X, customer_id index list)
        """
        # Calculate features per customer
        agg_funcs = {
            'total_price': ['sum', 'mean'],
            'transaction_id': 'nunique',
            'product_id': 'nunique',
            'is_returned': 'mean',
            'transaction_date': 'max'
        }
        
        cust_df = df.groupby('customer_id').agg(agg_funcs)
        cust_df.columns = ['total_spend', 'avg_order_value', 'num_transactions', 
                           'num_unique_products', 'return_rate', 'last_purchase_date']
                           
        # Calculate recency
        max_date = df['transaction_date'].max()
        cust_df['recency_days'] = (max_date - cust_df['last_purchase_date']).dt.days
        cust_df = cust_df.drop('last_purchase_date', axis=1)
        
        # Add categorical features (most frequent)
        cats = df.groupby('customer_id')[['product_category', 'payment_method']].agg(lambda x: x.value_counts().index[0])
        cats.columns = ['favorite_category', 'preferred_payment']
        
        cust_df = cust_df.join(cats)
        
        # One-hot encode categoricals
        cust_df = pd.get_dummies(cust_df, columns=['favorite_category', 'preferred_payment'])
        
        # Scale numericals
        num_cols = ['total_spend', 'avg_order_value', 'num_transactions', 'num_unique_products', 'return_rate', 'recency_days']
        cust_df[num_cols] = self.scaler.fit_transform(cust_df[num_cols])
        
        return cust_df.values, cust_df.index.tolist()

    def encode_products(self, df: pd.DataFrame) -> tuple:
        """
        Label encode product_id and product_name.
        
        Args:
            df (pd.DataFrame): Transaction DataFrame.
            
        Returns:
            tuple: (product_id_mapping dict, product_name_mapping dict)
        """
        unique_products = df[['product_id', 'product_name']].drop_duplicates()
        
        self.product_le.fit(unique_products['product_id'])
        self.product_name_le.fit(unique_products['product_name'])
        
        id_map = dict(zip(unique_products['product_id'], self.product_le.transform(unique_products['product_id'])))
        name_map = dict(zip(unique_products['product_name'], self.product_name_le.transform(unique_products['product_name'])))
        
        return id_map, name_map
