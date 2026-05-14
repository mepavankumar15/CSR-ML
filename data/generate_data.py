import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_data(num_transactions=50000, seed=42):
    """
    Generate a realistic synthetic e-commerce transaction dataset for cross-sell modeling.

    Columns to generate:
    - customer_id        : Unique customer identifier (C0001 - C5000)
    - transaction_id     : Unique transaction ID (T000001 - T050000)
    - transaction_date   : Random dates between 2021-01-01 and 2024-12-31
    - product_id         : Product purchased (P001 - P050)
    - product_name       : Mapped product names across 10 categories
    - product_category   : One of 10 categories (Electronics, Clothing, Books, etc.)
    - quantity           : 1–5 units
    - unit_price         : Price based on category (Electronics: 200-2000, Books: 10-80, etc.)
    - total_price        : quantity * unit_price
    - customer_age       : 18–70
    - customer_gender    : Male/Female
    - customer_segment   : Premium/Standard/Budget based on total spend
    - city               : One of 20 Indian cities
    - payment_method     : Credit Card / Debit Card / UPI / Net Banking / COD
    - is_returned        : 5% return rate
    """
    np.random.seed(seed)
    random.seed(seed)
    
    print("Generating synthetic data. This may take a few seconds...")

    # Configuration
    NUM_CUSTOMERS = 5000
    NUM_PRODUCTS = 50
    START_DATE = datetime(2021, 1, 1)
    END_DATE = datetime(2024, 12, 31)
    
    # 1. Generate Customers
    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, NUM_CUSTOMERS + 1)]
    ages = np.random.randint(18, 71, size=NUM_CUSTOMERS)
    genders = np.random.choice(['Male', 'Female'], size=NUM_CUSTOMERS, p=[0.5, 0.5])
    
    indian_cities = [
        "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Ahmedabad", "Chennai",
        "Kolkata", "Surat", "Pune", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
        "Indore", "Thane", "Bhopal", "Visakhapatnam", "Pimpri-Chinchwad", "Patna", "Vadodara"
    ]
    cities = np.random.choice(indian_cities, size=NUM_CUSTOMERS)
    
    # Map customers to their attributes
    customer_data = {cid: {'age': age, 'gender': gender, 'city': city} 
                     for cid, age, gender, city in zip(customer_ids, ages, genders, cities)}

    # 2. Generate Products
    categories = [
        "Electronics", "Clothing", "Books", "Home & Kitchen", "Beauty",
        "Sports", "Toys", "Groceries", "Health", "Automotive"
    ]
    
    # Price ranges per category
    price_ranges = {
        "Electronics": (200, 2000),
        "Clothing": (20, 150),
        "Books": (10, 80),
        "Home & Kitchen": (30, 300),
        "Beauty": (15, 100),
        "Sports": (25, 250),
        "Toys": (10, 120),
        "Groceries": (5, 50),
        "Health": (10, 80),
        "Automotive": (50, 500)
    }
    
    product_ids = [f"P{str(i).zfill(3)}" for i in range(1, NUM_PRODUCTS + 1)]
    product_data = {}
    
    for pid in product_ids:
        category = np.random.choice(categories)
        min_p, max_p = price_ranges[category]
        unit_price = round(np.random.uniform(min_p, max_p), 2)
        name = f"{category} Item {pid}"
        product_data[pid] = {'name': name, 'category': category, 'price': unit_price}

    # 3. Generate Transactions
    # Some customers buy more frequently
    customer_purchase_probs = np.random.dirichlet(np.ones(NUM_CUSTOMERS) * 0.1)
    
    # Generate dates
    days_between = (END_DATE - START_DATE).days
    random_days = np.random.randint(0, days_between, size=num_transactions)
    dates = [START_DATE + timedelta(days=int(d)) for d in random_days]
    dates.sort() # chronological
    
    transaction_ids = [f"T{str(i).zfill(6)}" for i in range(1, num_transactions + 1)]
    
    data = []
    
    # Category affinities based on age/gender (simplified simulation)
    payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "COD"]
    
    for i in range(num_transactions):
        cid = np.random.choice(customer_ids, p=customer_purchase_probs)
        pid = np.random.choice(product_ids) # Can add affinity logic here
        
        c_info = customer_data[cid]
        p_info = product_data[pid]
        
        qty = np.random.randint(1, 6)
        total_price = round(qty * p_info['price'], 2)
        
        is_returned = 1 if np.random.rand() < 0.05 else 0
        payment = np.random.choice(payment_methods, p=[0.3, 0.2, 0.35, 0.1, 0.05])
        
        data.append({
            'customer_id': cid,
            'transaction_id': transaction_ids[i],
            'transaction_date': dates[i].strftime('%Y-%m-%d'),
            'product_id': pid,
            'product_name': p_info['name'],
            'product_category': p_info['category'],
            'quantity': qty,
            'unit_price': p_info['price'],
            'total_price': total_price,
            'customer_age': c_info['age'],
            'customer_gender': c_info['gender'],
            'city': c_info['city'],
            'payment_method': payment,
            'is_returned': is_returned
        })
        
    df = pd.DataFrame(data)
    
    # Calculate Customer Segment based on total spend
    customer_spend = df.groupby('customer_id')['total_price'].sum()
    p75 = customer_spend.quantile(0.75)
    p25 = customer_spend.quantile(0.25)
    
    def get_segment(spend):
        if spend >= p75: return "Premium"
        elif spend <= p25: return "Budget"
        else: return "Standard"
        
    spend_map = customer_spend.apply(get_segment).to_dict()
    df['customer_segment'] = df['customer_id'].map(spend_map)
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_transactions.csv')
    df.to_csv(output_file, index=False)
    print(f"Dataset generated successfully with {len(df)} rows.")
    print(f"Saved to: {output_file}")
    
if __name__ == "__main__":
    generate_data(50000)
