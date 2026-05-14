import os
import pandas as pd
from src.recommender import CrossSellRecommender

def format_currency(amount: float) -> str:
    """Format float into Indian Rupees string."""
    return f"₹{amount:,.2f}"

def get_category_icon(category: str) -> str:
    """Return an emoji representation for a given product category."""
    icons = {
        "Electronics": "💻",
        "Clothing": "👕",
        "Books": "📚",
        "Home & Kitchen": "🍳",
        "Beauty": "💄",
        "Sports": "⚽",
        "Toys": "🧸",
        "Groceries": "🛒",
        "Health": "💊",
        "Automotive": "🚗"
    }
    return icons.get(category, "📦")

def create_product_card_data(product_id: str, details_dict: dict) -> dict:
    """Prepare a structured dictionary for the Streamlit UI."""
    return {
        "id": product_id,
        "name": details_dict.get('product_name', 'Unknown'),
        "category": details_dict.get('product_category', 'Unknown'),
        "icon": get_category_icon(details_dict.get('product_category', '')),
        "price": format_currency(details_dict.get('unit_price', 0))
    }

def color_by_score(score: float) -> str:
    """Return a hex color (green gradient) based on a normalized score (0 to 1)."""
    # From dark green to bright green
    # 0 = #004d00, 1 = #00ff00
    green_val = int(77 + (score * 178)) # 77 is 4d in hex, 255 is ff
    green_val = min(255, max(0, green_val))
    return f"#00{green_val:02x}00"

def truncate_name(name: str, max_len: int = 30) -> str:
    """Truncate a string to max_len characters."""
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + "..."

def load_or_train_models(data_path: str, model_path: str, force_retrain: bool = False) -> CrossSellRecommender:
    """
    Load existing models or train a fresh instance.
    
    Args:
        data_path (str): Path to transaction CSV.
        model_path (str): Path to save/load model pkl.
        force_retrain (bool): Whether to skip loading and force training.
        
    Returns:
        CrossSellRecommender: The loaded or freshly trained recommender instance.
    """
    if not force_retrain and os.path.exists(model_path):
        print(f"Loading existing models from {model_path}...")
        try:
            return CrossSellRecommender.load_models(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}. Retraining...")
            
    # Retrain
    print("Training new models...")
    df = pd.read_csv(data_path)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    recommender = CrossSellRecommender()
    recommender.fit(df)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    recommender.save_models(model_path)
    
    return recommender
