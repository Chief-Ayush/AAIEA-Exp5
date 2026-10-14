"""Synthetic-but-realistic product catalog + three simulated marketplaces.

* ``canonical`` catalog  = ground truth (also the authoritative "product detail" source).
* Each marketplace returns a *noisy partial view*: coverage < 100 %, small price drift, rating disagreements,
  occasional missing fields.  This is what makes synchronization / conflict resolution a real problem.
All brands are fictional.  Everything is deterministic for a given seed.
"""
from __future__ import annotations

import hashlib
import math
import random
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import PATHS, SOURCES

COLUMNS = ["product_id", "product_name", "category", "brand", "price", "rating", "reviews", "ram_gb", "storage_gb",
           "battery_hours", "screen_inches", "weight_kg", "warranty_months", "availability", "features"]

# the ten products of the lab notebook are kept verbatim so the lab-manual query gives the familiar answer
SEED_PRODUCTS = [
    ("P001", "TechPro Laptop X1", "Laptop", "TechPro", 64999, 4.5, 1850, 16, 512, 9, 14.0, 1.4, 12, "Available", "SSD/Backlit Keyboard/WiFi 6/Fingerprint"),
    ("P002", "MegaBook Pro 15", "Laptop", "MegaBook", 68999, 4.7, 2450, 16, 1024, 11, 15.6, 1.7, 12, "Available", "SSD/Backlit Keyboard/WiFi 6/Fingerprint"),
    ("P003", "UltraTech Air 14", "Laptop", "UltraTech", 59999, 4.3, 1320, 16, 512, 8, 14.0, 1.3, 12, "Available", "SSD/WiFi 6"),
    ("P004", "PowerMax Gaming 15", "Laptop", "PowerMax", 74999, 4.6, 2100, 16, 1024, 7, 15.6, 2.2, 12, "Available", "SSD/Dedicated GPU/Backlit Keyboard"),
    ("P005", "BudgetBook 14", "Laptop", "BudgetBook", 44999, 4.1, 980, 8, 512, 7, 14.0, 1.6, 12, "Available", "SSD/WiFi 5"),
    ("P006", "SmartPhone Z Pro", "Smartphone", "SmartTech", 29999, 4.6, 3200, 12, 256, 14, 6.6, 0.19, 12, "Available", "5G/AMOLED/Fast Charging"),
    ("P007", "PhoneMax Ultra", "Smartphone", "PhoneMax", 27999, 4.4, 2800, 8, 256, 13, 6.5, 0.2, 12, "Available", "5G/AMOLED/Fast Charging"),
    ("P008", "TabMaster Plus", "Tablet", "TabMaster", 24999, 4.5, 1650, 8, 256, 12, 11.0, 0.5, 12, "Available", "2K Display/5G/Fast Charging"),
    ("P009", "UltraSound Pro", "Headphones", "UltraSound", 8999, 4.6, 4100, 0, 0, 30, 0, 0.26, 12, "Available", "ANC/Bluetooth/Spatial Audio"),
    ("P010", "SoundMax Wireless", "Headphones", "SoundMax", 5999, 4.3, 2900, 0, 0, 25, 0, 0.25, 12, "Available", "ANC/Bluetooth/Fast Charging"),
]

# (price multiplier, base rating) per brand
BRANDS = {
    "Laptop": {"TechPro": (1.0, 4.3), "MegaBook": (1.12, 4.5), "UltraTech": (0.95, 4.3), "PowerMax": (1.2, 4.4), "BudgetBook": (0.72, 4.0),
               "NovaCore": (1.05, 4.4), "Zephyra": (1.15, 4.5), "AeroBook": (1.1, 4.4), "Lumina": (0.98, 4.2), "CoreEdge": (0.9, 4.1),
               "Orbitron": (1.25, 4.5), "Vertex": (0.85, 4.2)},
    "Smartphone": {"SmartTech": (1.0, 4.4), "PhoneMax": (0.95, 4.3), "Nexon": (1.1, 4.5), "Quanta": (0.85, 4.1), "Zenith": (1.2, 4.5),
                   "Pixelio": (1.05, 4.4), "Lavo": (0.7, 4.0), "Vibe": (0.8, 4.2), "Aurix": (1.3, 4.6)},
    "Tablet": {"TabMaster": (1.0, 4.4), "SlateOne": (0.9, 4.2), "Lumina": (1.0, 4.3), "Nexon": (1.15, 4.5), "Quanta": (0.8, 4.1), "AeroBook": (1.1, 4.4)},
    "Headphones": {"UltraSound": (1.1, 4.5), "SoundMax": (0.8, 4.2), "Auralis": (1.25, 4.6), "BassBeat": (0.7, 4.0), "Echoline": (1.0, 4.4),
                   "Quanta": (0.85, 4.2), "Zenith": (1.2, 4.5), "Vibe": (0.75, 4.1)},
    "Smartwatch": {"FitPulse": (0.9, 4.2), "ChronoX": (1.1, 4.4), "Nexon": (1.15, 4.5), "Zenith": (1.2, 4.5), "Vibe": (0.7, 4.0),
                   "AuraFit": (1.0, 4.3), "Quanta": (0.8, 4.1)},
}
SERIES = {"Laptop": ["Air", "Pro", "Ultra", "Edge", "Flex", "Book", "Gaming", "Studio", "Lite", "Max"],
          "Smartphone": ["Z", "X", "Neo", "Prime", "Lite", "Ultra", "Edge", "Note", "Pro", "Max"],
          "Tablet": ["Plus", "Air", "Pro", "Lite", "Mini", "Max", "Note"],
          "Headphones": ["Pro", "Wireless", "Buds", "Studio", "Beat", "Elite", "Bass", "Air", "Sport"],
          "Smartwatch": ["Watch", "Fit", "Active", "Pro", "Sport", "Lite", "Classic", "Ultra"]}
FEATURE_POOL = {"Laptop": ["Backlit Keyboard", "WiFi 6", "Fingerprint", "Touchscreen", "Dedicated GPU", "Thunderbolt", "OLED", "Numpad"],
                "Smartphone": ["5G", "AMOLED", "Fast Charging", "Wireless Charging", "Dual SIM", "120Hz Display", "NFC", "Water Resistant"],
                "Tablet": ["2K Display", "5G", "Fast Charging", "Stylus Support", "Keyboard Support", "AMOLED", "Dual Speakers"],
                "Headphones": ["ANC", "Bluetooth", "Spatial Audio", "Fast Charging", "Multipoint", "Water Resistant", "Wireless Charging"],
                "Smartwatch": ["GPS", "AMOLED", "SpO2", "NFC", "Heart Rate", "Water Resistant", "Fast Charging"]}
COUNTS = {"Laptop": 42, "Smartphone": 38, "Tablet": 22, "Headphones": 28, "Smartwatch": 24}


def _price_round(p: float) -> int:
    return int(round(p / 1000.0) * 1000 - 1) if p >= 8000 else int(round(p / 100.0) * 100 - 1)


def _make(cat: str, rng: random.Random) -> dict:
    brand = rng.choice(list(BRANDS[cat]))
    mult, base_rating = BRANDS[cat][brand]
    gpu = False
    if cat == "Laptop":
        ram = rng.choice([8, 8, 16, 16, 16, 32])
        storage = rng.choice([256, 512, 512, 512, 1024, 1024, 2048])
        screen = rng.choice([13.3, 14.0, 14.0, 15.6, 15.6, 16.0])
        gpu = rng.random() < 0.28
        battery = round(rng.uniform(5.5, 15.5) - (3 if gpu else 0), 1)
        weight = round(rng.uniform(1.1, 1.8) + (0.45 if gpu else 0), 2)
        price = (17000 + ram * 1250 + storage * 14 + (17000 if gpu else 0) + (screen - 13) * 1500) * mult
    elif cat == "Smartphone":
        ram = rng.choice([4, 6, 8, 8, 12, 12, 16])
        storage = rng.choice([64, 128, 128, 256, 256, 512])
        screen = round(rng.uniform(6.1, 6.9), 1)
        battery = round(rng.uniform(15, 34), 1)
        weight = round(rng.uniform(0.17, 0.23), 2)
        price = (6500 + ram * 1250 + storage * 38) * mult
    elif cat == "Tablet":
        ram = rng.choice([4, 6, 8, 8, 12])
        storage = rng.choice([64, 128, 256, 256, 512])
        screen = rng.choice([8.7, 10.1, 10.9, 11.0, 12.4])
        battery = round(rng.uniform(9, 20), 1)
        weight = round(rng.uniform(0.3, 0.65), 2)
        price = (7500 + ram * 1000 + storage * 33 + (screen - 8) * 2000) * mult
    elif cat == "Headphones":
        ram = storage = 0
        screen = 0.0
        battery = round(rng.uniform(14, 60), 1)
        weight = round(rng.uniform(0.05, 0.30), 2)
        price = (1200 + battery * 85 + rng.choice([0, 1500, 3200, 5200])) * mult
    else:  # Smartwatch
        ram = storage = 0
        screen = round(rng.uniform(1.2, 1.9), 1)
        battery = round(rng.uniform(30, 300), 1)
        weight = round(rng.uniform(0.03, 0.07), 2)
        price = (2200 + battery * 18 + rng.choice([0, 1500, 4000])) * mult
    price = _price_round(price * rng.uniform(0.93, 1.07))
    rating = base_rating + 0.12 * math.tanh((price - 30000) / 40000) + rng.gauss(0, 0.22)
    rating = round(min(4.9, max(3.4, rating)), 1)
    reviews = int(math.exp(rng.uniform(math.log(140), math.log(9500))))
    feats = rng.sample(FEATURE_POOL[cat], k=rng.randint(2, min(5, len(FEATURE_POOL[cat]))))
    if gpu and "Dedicated GPU" not in feats:
        feats.append("Dedicated GPU")
    if cat == "Laptop" and not gpu:
        feats = [f for f in feats if f != "Dedicated GPU"]
    av = rng.choices(["Available", "Limited Stock", "Out of Stock"], weights=[85, 8, 7])[0]
    return dict(category=cat, brand=brand, price=price, rating=rating, reviews=reviews, ram_gb=ram, storage_gb=storage,
                battery_hours=battery, screen_inches=screen, weight_kg=weight, warranty_months=rng.choice([12, 12, 12, 24]),
                availability=av, features="SSD/" + "/".join(feats) if cat == "Laptop" else "/".join(feats))


def generate_canonical(seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [dict(zip(COLUMNS, r)) for r in SEED_PRODUCTS]
    names = {r["product_name"] for r in rows}
    n = len(rows)
    have = {c: sum(1 for r in rows if r["category"] == c) for c in COUNTS}
    for cat, total in COUNTS.items():
        for _ in range(total - have[cat]):
            for _try in range(50):
                rec = _make(cat, rng)
                name = f"{rec['brand']} {rng.choice(SERIES[cat])} {rng.choice([10, 12, 14, 15, 16, 20, 30, 40, 50, 55, 60, 70, 100])}"
                if name not in names:
                    break
            names.add(name)
            n += 1
            rows.append({"product_id": f"P{n:03d}", "product_name": name, **rec})
    return pd.DataFrame(rows)[COLUMNS]


def build_sources(canonical: pd.DataFrame, seed: int = 7, poison: dict | None = None) -> dict[str, pd.DataFrame]:
    """Noisy marketplace views of the canonical catalog.  ``poison`` = {source: {product_id: {field: value}}} (security tests)."""
    out: dict[str, list[dict]] = {s: [] for s in SOURCES}
    for _, row in canonical.iterrows():
        pid = row["product_id"]
        present = []
        for s, meta in SOURCES.items():
            rng = random.Random(f"{seed}|{s}|{pid}")
            if rng.random() < meta["coverage"]:
                present.append((s, rng))
        if not present:  # every product must be sold somewhere
            s = max(SOURCES, key=lambda k: SOURCES[k]["trust"])
            present = [(s, random.Random(f"{seed}|{s}|{pid}|fallback"))]
        rs = random.Random(f"{seed}|stale|{pid}")
        stale = rs.uniform(0.90, 0.97) if rs.random() < 0.035 else (rs.uniform(1.03, 1.07) if rs.random() < 0.02 else 1.0)   # upstream price change not yet propagated
        for s, rng in present:
            rec = row.to_dict()
            rec["price"] = int(round(row["price"] * stale)) if stale != 1.0 else row["price"]
            r = rng.random()
            rec["rating"] = row["rating"] if r < 0.55 else round(min(5.0, max(1.0, row["rating"] + rng.choice([-0.1, 0.1]))), 1) if r < 0.82 \
                else round(min(5.0, max(1.0, row["rating"] + rng.choice([-0.4, -0.3, -0.2, 0.2, 0.3, 0.4]))), 1)
            if SOURCES[s]["trust"] > 0.85 and r >= 0.82:      # high-trust sources are less noisy
                rec["rating"] = row["rating"]
            pj = rng.random()
            if pj > 0.7:
                rec["price"] = int(round(rec["price"] * (1 + rng.uniform(-0.03, 0.045)) / 10.0) * 10 - 1)
            if s == "BrandOutlet" and pj > 0.9:
                rec["price"] = int(rec["price"] * 0.96)
            rec["reviews"] = max(10, int(row["reviews"] * rng.uniform(0.55, 1.0)))
            if rng.random() < 0.03:
                rec["rating"] = None                           # missing data
            if rng.random() < 0.015:
                rec["battery_hours"] = None
            if row["availability"] == "Out of Stock" and rng.random() < 0.4:
                rec["availability"] = "Available"              # stale stock information
            rec["source"] = s
            rec["source_trust"] = SOURCES[s]["trust"]
            out[s].append(rec)
    if poison:
        for s, items in poison.items():
            for rec in out[s]:
                if rec["product_id"] in items:
                    rec.update(items[rec["product_id"]])
    return {s: pd.DataFrame(v) for s, v in out.items()}


def save_catalog(canonical: pd.DataFrame, seed: int = 7) -> None:
    PATHS.ensure()
    canonical.to_csv(PATHS.catalog_csv, index=False)
    for s, df in build_sources(canonical, seed).items():
        df.to_csv(PATHS.sources / f"{s}.csv", index=False)


def load_canonical() -> pd.DataFrame:
    if not PATHS.catalog_csv.exists():
        save_catalog(generate_canonical())
    df = pd.read_csv(PATHS.catalog_csv)
    return df


REQUIRED_COLUMNS = ["product_id", "product_name", "category", "brand", "price", "rating"]


def validate_catalog(df: pd.DataFrame) -> list[str]:
    """Dataset verification (lab Step 4)."""
    issues = []
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            issues.append(f"missing column: {c}")
    if issues:
        return issues
    if df["product_id"].duplicated().any():
        issues.append("duplicate product_id values")
    if (df["price"] <= 0).any():
        issues.append("non-positive prices")
    if not df["rating"].between(0, 5).all():
        issues.append("ratings outside 0..5")
    for c in ("ram_gb", "storage_gb", "battery_hours"):
        if c in df.columns and (df[c].dropna() < 0).any():
            issues.append(f"negative values in {c}")
    return issues


def normalize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {"reviews": 100, "ram_gb": 0, "storage_gb": 0, "battery_hours": np.nan, "screen_inches": 0.0, "weight_kg": np.nan,
                "warranty_months": 12, "availability": "Available", "features": ""}
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d
    df["features"] = df["features"].fillna("").astype(str)
    return df[COLUMNS]


class CatalogStore:
    """Holds the canonical catalog + the marketplace views (rebuilt when a custom catalog is loaded)."""

    def __init__(self, canonical: pd.DataFrame | None = None, seed: int = 7, poison: dict | None = None):
        self.seed = seed
        self.canonical = normalize_catalog(canonical if canonical is not None else load_canonical())
        self.sources = build_sources(self.canonical, seed, poison)
        self.by_id = self.canonical.set_index("product_id", drop=False)

    @property
    def categories(self) -> list[str]:
        return sorted(self.canonical["category"].unique())

    @property
    def brands(self) -> list[str]:
        return sorted(self.canonical["brand"].unique())

    def product_names(self) -> dict[str, str]:
        return dict(zip(self.canonical["product_name"], self.canonical["product_id"]))

    def detail(self, pid: str) -> dict | None:
        if pid not in self.by_id.index:
            return None
        r = self.by_id.loc[pid].to_dict()
        return {k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()}

    def stats(self) -> dict:
        c = self.canonical
        return {"products": len(c), "categories": c["category"].nunique(), "brands": c["brand"].nunique(),
                "source_records": {s: len(d) for s, d in self.sources.items()},
                "avg_rating": round(float(c["rating"].mean()), 2), "price_min": int(c["price"].min()), "price_max": int(c["price"].max())}


@lru_cache(maxsize=1)
def default_store() -> CatalogStore:
    return CatalogStore()


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:10]
