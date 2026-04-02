from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = "2018-03-01"
EVENT_DATE = pd.Timestamp("2023-03-10")


UNIVERSE = [
    {"ticker": "USB", "name": "U.S. Bancorp", "group": "Treatment Big"},
    {"ticker": "PNC", "name": "PNC Financial Services", "group": "Treatment Big"},
    {"ticker": "TFC", "name": "Truist Financial", "group": "Treatment Big"},
    {"ticker": "FITB", "name": "Fifth Third Bancorp", "group": "Treatment Big"},
    {"ticker": "CFG", "name": "Citizens Financial Group", "group": "Treatment Big"},
    {"ticker": "MTB", "name": "M&T Bank", "group": "Treatment Big"},
    {"ticker": "HBAN", "name": "Huntington Bancshares", "group": "Treatment Big"},
    {"ticker": "ZION", "name": "Zions Bancorporation", "group": "Treatment Small"},
    {"ticker": "CMA", "name": "Comerica", "group": "Treatment Small"},
    {"ticker": "KEY", "name": "KeyCorp", "group": "Treatment Small"},
    {
        "ticker": "WAL",
        "name": "Western Alliance Bancorporation",
        "group": "Treatment Small",
    },
    {"ticker": "BOKF", "name": "BOK Financial", "group": "Treatment Small"},
    {"ticker": "FHN", "name": "First Horizon", "group": "Treatment Small"},
    {"ticker": "PB", "name": "Prosperity Bancshares", "group": "Treatment Small"},
    {"ticker": "EWBC", "name": "East West Bancorp", "group": "Treatment Small"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "group": "Control Big"},
    {"ticker": "BAC", "name": "Bank of America", "group": "Control Big"},
    {"ticker": "WFC", "name": "Wells Fargo", "group": "Control Big"},
    {"ticker": "C", "name": "Citigroup", "group": "Control Big"},
    {"ticker": "MS", "name": "Morgan Stanley", "group": "Control Big"},
    {"ticker": "GS", "name": "Goldman Sachs", "group": "Control Big"},
    {"ticker": "BX", "name": "Blackstone", "group": "Control Big"},
    {"ticker": "BLK", "name": "BlackRock", "group": "Control Big"},
    {"ticker": "COF", "name": "Capital One Financial", "group": "Control Small"},
    {"ticker": "SYF", "name": "Synchrony Financial", "group": "Control Small"},
    {"ticker": "ALL", "name": "Allstate", "group": "Control Small"},
    {"ticker": "TRV", "name": "Travelers", "group": "Control Small"},
    {"ticker": "PRU", "name": "Prudential Financial", "group": "Control Small"},
    {"ticker": "MET", "name": "MetLife", "group": "Control Small"},
    {"ticker": "HIG", "name": "The Hartford", "group": "Control Small"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "group": "Benchmark"},
]


def download_prices(tickers: list[str]) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=end_date.strftime("%Y-%m-%d"),
        interval="1mo",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("No price data returned by yfinance.")

    frames: list[pd.DataFrame] = []

    if isinstance(raw.columns, pd.MultiIndex):
        bulk_df = (
            raw.stack(level=0, future_stack=True)
            .rename_axis(index=["Date", "Ticker"])
            .reset_index()
        )
        bulk_df["Date"] = pd.to_datetime(bulk_df["Date"])
        frames.append(bulk_df)
        available_tickers = set(bulk_df["Ticker"].unique())
    else:
        available_tickers = set()

    missing_tickers = [ticker for ticker in tickers if ticker not in available_tickers]

    for ticker in missing_tickers:
        single = yf.download(
            tickers=ticker,
            start=START_DATE,
            end=end_date.strftime("%Y-%m-%d"),
            interval="1mo",
            auto_adjust=False,
            progress=False,
            threads=False,
            actions=False,
        )
        if single.empty:
            continue
        single = single.reset_index()
        single["Ticker"] = ticker
        frames.append(single)

    if not frames:
        raise RuntimeError("No price data returned by yfinance.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["Date"] = pd.to_datetime(long_df["Date"])

    meta = pd.DataFrame(UNIVERSE).rename(
        columns={"ticker": "Ticker", "name": "Name", "group": "Group"}
    )
    long_df = long_df.merge(meta, on="Ticker", how="left")
    long_df = long_df[
        [
            "Date",
            "Ticker",
            "Name",
            "Group",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        ]
    ]

    long_df = long_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return long_df


def split_event_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = df[df["Date"] < EVENT_DATE].copy().reset_index(drop=True)
    after = df[df["Date"] >= EVENT_DATE].copy().reset_index(drop=True)
    return before, after


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "event_study_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [item["ticker"] for item in UNIVERSE]
    full_data = download_prices(tickers)
    before_data, after_data = split_event_windows(full_data)

    before_path = output_dir / "prices_before_2023-03-10.csv"
    after_path = output_dir / "prices_after_2023-03-10.csv"
    universe_path = output_dir / "ticker_universe.csv"

    before_data.to_csv(before_path, index=False)
    after_data.to_csv(after_path, index=False)
    pd.DataFrame(UNIVERSE).to_csv(universe_path, index=False)

    print("Downloaded price data successfully.")
    print(f"Universe size: {len(UNIVERSE)} tickers")
    print(f"Before-event rows: {len(before_data):,} -> {before_path}")
    print(f"After-event rows: {len(after_data):,} -> {after_path}")
    print(f"Ticker universe -> {universe_path}")
    print("\nDate coverage:")
    print(
        f"  Before: {before_data['Date'].min().date()} to {before_data['Date'].max().date()}"
    )
    print(
        f"  After : {after_data['Date'].min().date()} to {after_data['Date'].max().date()}"
    )
    print("\nPreview (before-event):")
    print(before_data.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
