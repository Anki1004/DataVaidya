"""Generate 5 demo datasets for DataVaidya, each seeded with intentional quality issues.

Run from project root:
    python make_demo.py

Datasets produced (all seeded random_state=42):
1. titanic.csv                - classic Titanic with extra missingness + duplicates
2. iris.csv                   - Iris + mixed-dtype column + constant column
3. census_india_2011.csv      - district-level demographics + class imbalance
4. mumbai_real_estate.csv     - listings + inconsistent dates + outliers
5. retail_transactions.csv    - Indian retail with PII fields (emails, phones)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "samples"
SEED = 42

def make_titanic() -> pd.DataFrame:
    np.random.seed(SEED); fake = Faker(); Faker.seed(SEED)
    n = 891
    df = pd.DataFrame({
        "PassengerId": range(1, n+1),
        "Survived": np.random.choice([0,1], n, p=[0.62,0.38]),
        "Pclass": np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
        "Name": [fake.name() for _ in range(n)],
        "Sex": np.random.choice(["male","female"], n, p=[0.65,0.35]),
        "Age": np.random.normal(29.7, 14.5, n).round(1),
        "SibSp": np.random.choice([0,1,2,3,4,5,8], n, p=[.68,.23,.03,.02,.02,.01,.01]),
        "Parch": np.random.choice([0,1,2,3,4,5,6], n, p=[.76,.13,.08,.01,.01,.005,.005]),
        "Ticket": [fake.bothify("?####??") for _ in range(n)],
        "Fare": np.abs(np.random.lognormal(2.9, 1.0, n)).round(2),
        "Cabin": [fake.bothify("?##") if np.random.rand()>0.77 else np.nan for _ in range(n)],
        "Embarked": np.random.choice(["S","C","Q"], n, p=[0.72,0.19,0.09]),
    })
    df.loc[np.random.choice(n, 220, replace=False), "Age"] = np.nan
    df.loc[np.random.choice(n, 2, replace=False), "Embarked"] = np.nan
    idx = np.random.choice(n, 30, replace=False)
    df.loc[idx[:10], "Sex"] = "Male"
    df.loc[idx[10:20], "Sex"] = "M"
    df = pd.concat([df, df.sample(5, random_state=SEED)], ignore_index=True)
    return df

def make_iris() -> pd.DataFrame:
    np.random.seed(SEED)
    n = 600
    species = np.repeat(["setosa","versicolor","virginica"], n//3)
    df = pd.DataFrame({
        "sepal_length": np.concatenate([np.random.normal(5.0, 0.5, n//3),
                                        np.random.normal(5.9, 0.5, n//3),
                                        np.random.normal(6.6, 0.5, n//3)]).round(2),
        "sepal_width":  np.concatenate([np.random.normal(3.4, 0.3, n//3),
                                        np.random.normal(2.8, 0.3, n//3),
                                        np.random.normal(3.0, 0.3, n//3)]).round(2),
        "petal_length": np.concatenate([np.random.normal(1.5, 0.4, n//3),
                                        np.random.normal(4.3, 0.4, n//3),
                                        np.random.normal(5.6, 0.4, n//3)]).round(2),
        "petal_width":  np.concatenate([np.random.normal(0.2, 0.2, n//3),
                                        np.random.normal(1.3, 0.2, n//3),
                                        np.random.normal(2.0, 0.2, n//3)]).round(2),
        "species": species,
        "dataset_version": "v1.0",
        "id": [f"IR-{i:05d}" for i in range(n)],
    })
    mix_idx = np.random.choice(n, 48, replace=False)
    df["petal_length"] = df["petal_length"].astype(object)
    df.loc[mix_idx, "petal_length"] = df.loc[mix_idx, "petal_length"].astype(str) + " cm"
    df.loc[np.random.choice(n, 4, replace=False), "sepal_length"] = 99.9
    df.loc[np.random.choice(n, 30, replace=False), "petal_width"] = np.nan
    df = pd.concat([df, df.sample(3, random_state=SEED)], ignore_index=True)
    return df

def make_census() -> pd.DataFrame:
    np.random.seed(SEED); fake = Faker("en_IN"); Faker.seed(SEED)
    states = ["Uttar Pradesh","Maharashtra","Bihar","West Bengal","Madhya Pradesh",
              "Tamil Nadu","Rajasthan","Karnataka","Gujarat","Andhra Pradesh"]
    n = 640
    df = pd.DataFrame({
        "state": np.random.choice(states, n),
        "district": [f"{fake.city()}_{i}" for i in range(n)],
        "total_population": np.random.lognormal(13.5, 0.9, n).astype(int),
        "literacy_rate": np.clip(np.random.normal(73, 12, n), 30, 99).round(1),
        "sex_ratio": np.random.normal(940, 40, n).astype(int),
        "urban_pct": np.clip(np.random.normal(31, 18, n), 0, 100).round(1),
        "is_metro": np.random.choice([True, False], n, p=[0.03, 0.97]),
    })
    up_mask = df["state"] == "Uttar Pradesh"
    up_idx = df[up_mask].sample(min(20, up_mask.sum()), random_state=1).index
    df.loc[up_idx[:7], "state"] = "uttar pradesh"
    df.loc[up_idx[7:14], "state"] = "U.P."
    df.loc[np.random.choice(n, 26, replace=False), "literacy_rate"] = np.nan
    df.loc[np.random.choice(n, 2, replace=False), "sex_ratio"] = [300, 1500]
    df = pd.concat([df, df.sample(6, random_state=3)], ignore_index=True)
    return df

def make_mumbai_re() -> pd.DataFrame:
    np.random.seed(SEED); fake = Faker(); Faker.seed(SEED)
    hoods = ["Bandra","Andheri","Powai","Dadar","Worli","Juhu","Goregaon","Borivali","Thane","Navi Mumbai"]
    n = 2500
    base = datetime(2024, 1, 1)
    dates = [base + timedelta(days=int(d)) for d in np.random.randint(0, 500, n)]
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%d-%b-%y"]
    df = pd.DataFrame({
        "neighborhood": np.random.choice(hoods, n),
        "bhk": np.random.choice([1,2,3,4,5], n, p=[.15,.4,.3,.1,.05]),
        "area_sqft": np.random.normal(900, 350, n).astype(int).clip(250, 4000),
        "price_inr": np.random.lognormal(16.2, 0.6, n).round(0),
        "age_years": np.random.randint(0, 40, n),
        "amenities": [", ".join(np.random.choice(["Gym","Pool","Parking","Security","Garden"],
                                                  np.random.randint(1,5), replace=False)) for _ in range(n)],
        "listed_on": [d.strftime(np.random.choice(fmts)) for d in dates],
    })
    mix = np.random.choice(n, 40, replace=False)
    df.loc[mix[:10], "neighborhood"] = "mumbai"
    df.loc[mix[10:20], "neighborhood"] = "MUMBAI"
    df.loc[mix[20:30], "neighborhood"] = "Bombay"
    df.loc[np.random.choice(n, 3, replace=False), "price_inr"] = [500000000, 600000000, 750000000]
    df.loc[np.random.choice(n, 250, replace=False), "area_sqft"] = np.nan
    df.loc[np.random.choice(n, 2, replace=False), "bhk"] = 99
    df = pd.concat([df, df.sample(8, random_state=11)], ignore_index=True)
    return df

def make_retail() -> pd.DataFrame:
    np.random.seed(SEED); fake = Faker("en_IN"); Faker.seed(SEED)
    cats = ["Electronics","Apparel","Grocery","Books","Home","Beauty"]
    n = 3000
    df = pd.DataFrame({
        "order_id": [f"ORD{i:07d}" for i in range(n)],
        "customer_id": [f"CUST{np.random.randint(1, 800):05d}" for _ in range(n)],
        "product_category": np.random.choice(cats, n, p=[.55, .15, .12, .08, .06, .04]),
        "quantity": np.random.choice([1,2,3,4,5], n, p=[.5, .25, .15, .07, .03]).astype(object),
        "price_inr": np.random.lognormal(6.8, 1.0, n).round(2),
        "transaction_date": pd.date_range("2024-01-01", periods=n, freq="2h"),
        "customer_email": [fake.email() for _ in range(n)],
        "customer_phone": [f"+91-9{np.random.randint(100000000, 999999999)}" for _ in range(n)],
    })
    word_idx = np.random.choice(n, 210, replace=False)
    words = ["one","two","three","four","five"]
    df.loc[word_idx, "quantity"] = [words[i % 5] for i in range(len(word_idx))]
    df.loc[np.random.choice(n, 180, replace=False), "price_inr"] = np.nan
    df.loc[np.random.choice(n, 4, replace=False), "price_inr"] = [1500000, 2200000, 1800000, 3100000]
    dup_ids = df["order_id"].sample(12, random_state=5).values
    df.loc[df.sample(12, random_state=6).index, "order_id"] = dup_ids
    return df

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        ("titanic.csv", make_titanic),
        ("iris.csv", make_iris),
        ("census_india_2011.csv", make_census),
        ("mumbai_real_estate.csv", make_mumbai_re),
        ("retail_transactions.csv", make_retail),
    ]
    for name, builder in builders:
        df = builder()
        df.to_csv(OUTPUT_DIR / name, index=False)
        print(f"  wrote {name:32s} {df.shape[0]:>5} rows x {df.shape[1]:>2} cols")
    print(f"All demo datasets written to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
