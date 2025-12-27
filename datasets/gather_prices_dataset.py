import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

OUTPUT_CSV = "datasets/prices.csv"

# Synthetic pricing parameters (€/MWh)
BUY_MARKUP = 30.0     # added to spot price
SELL_DISCOUNT = 10.0  # subtracted from spot price
MIN_SELL_PRICE = 5.0   # minimum sell price (€/MWh)

# Free Estonian Nord Pool price API (Elering)
BASE_URL = "https://dashboard.elering.ee/api/nps/price"
COUNTRY_CODE = "ee"

YEARS = [2023, 2024]

def fetch_prices(start_iso: str, end_iso: str) -> pd.DataFrame:
    params = {
        "interval": "hour",
        "start": start_iso,
        "end": end_iso
    }
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    data = r.json().get("data", [])
    return pd.DataFrame(data[COUNTRY_CODE])

def chunk_range(start: datetime, end: datetime, step_days=30):
    current = start
    while current < end:
        nxt = min(current + timedelta(days=step_days), end)
        yield current, nxt
        current = nxt


def download_data():
    all_chunks = []

    for year in YEARS:
        print(f"Fetching prices for {year}")
        start = datetime(year, 1, 1)
        end = datetime(year + 1, 1, 1)

        for s, e in tqdm(list(chunk_range(start, end))):
            df = fetch_prices(
                s.isoformat() + "Z",
                e.isoformat() + "Z"
            )
            if not df.empty:
                all_chunks.append(df)

    return pd.concat(all_chunks, ignore_index=True)

def parse_data(df):
    # Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s', utc=True)

    # Spot price (€/MWh)
    df["spot_price_eur_mwh"] = df["price"]

    # Synthetic buy & sell prices
    df["buy_price_eur_mwh"] = df["spot_price_eur_mwh"] + BUY_MARKUP
    df["sell_price_eur_mwh"] = (
        df["spot_price_eur_mwh"] - SELL_DISCOUNT
    ).clip(lower=MIN_SELL_PRICE)

    df["buy_price_eur_kwh"] = df["buy_price_eur_mwh"] / 1000
    df["sell_price_eur_kwh"] = df["sell_price_eur_mwh"] / 1000

    return (
        df[[
            "timestamp",
            "buy_price_eur_kwh",
            "sell_price_eur_kwh"
        ]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def save_data(df):
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved dataset to: {OUTPUT_CSV}")
    print(df.head())


def main():
    df = download_data()
    df = parse_data(df)
    save_data(df)


if __name__ == "__main__":
    main()