from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = "2017-01-01"
END_DATE = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
EVENT_DATE = pd.Timestamp("2023-08-08")


UNIVERSE = [
    {"ticker": "ISRG", "name": "Intuitive Surgical", "group": "Treatment Big"},
    {"ticker": "DXCM", "name": "DexCom", "group": "Treatment Big"},
    {"ticker": "RMD", "name": "ResMed", "group": "Treatment Big"},
    {"ticker": "PODD", "name": "Insulet", "group": "Treatment Big"},
    {"ticker": "MASI", "name": "Masimo", "group": "Treatment Big"},
    {"ticker": "MDT", "name": "Medtronic", "group": "Treatment Big"},
    {"ticker": "PEN", "name": "Penumbra", "group": "Treatment Big"},
    {"ticker": "INSP", "name": "Inspire Medical Systems", "group": "Treatment Big"},
    {"ticker": "TNDM", "name": "Tandem Diabetes Care", "group": "Treatment Small"},
    {"ticker": "OM", "name": "Outset Medical", "group": "Treatment Small"},
    {"ticker": "CNMD", "name": "CONMED", "group": "Treatment Small"},
    {"ticker": "EW", "name": "Edwards Lifesciences", "group": "Treatment Small"},
    {"ticker": "MMSI", "name": "Merit Medical Systems", "group": "Treatment Small"},
    {"ticker": "LMAT", "name": "LeMaitre Vascular", "group": "Treatment Small"},
    {"ticker": "ANGO", "name": "AngioDynamics", "group": "Treatment Small"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "group": "Control Big"},
    {"ticker": "TMO", "name": "Thermo Fisher Scientific", "group": "Control Big"},
    {"ticker": "DHR", "name": "Danaher", "group": "Control Big"},
    {"ticker": "ZTS", "name": "Zoetis", "group": "Control Big"},
    {"ticker": "IDXX", "name": "IDEXX Laboratories", "group": "Control Big"},
    {"ticker": "COO", "name": "Cooper Companies", "group": "Control Big"},
    {"ticker": "ALGN", "name": "Align Technology", "group": "Control Big"},
    {"ticker": "BSX", "name": "Boston Scientific", "group": "Control Big"},
    {"ticker": "XRAY", "name": "Dentsply Sirona", "group": "Control Small"},
    {"ticker": "HAE", "name": "Haemonetics", "group": "Control Small"},
    {"ticker": "CERT", "name": "Certara", "group": "Control Small"},
    {"ticker": "NEO", "name": "NeoGenomics", "group": "Control Small"},
    {"ticker": "CSTL", "name": "Castle Biosciences", "group": "Control Small"},
    {"ticker": "GH", "name": "Guardant Health", "group": "Control Small"},
    {"ticker": "STAA", "name": "STAAR Surgical", "group": "Control Small"},
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "group": "Benchmark"},
]


def download_prices(tickers: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if raw is None:
        raise RuntimeError("No price data returned by yfinance.")

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
            end=END_DATE,
            interval="1mo",
            auto_adjust=False,
            progress=False,
            threads=False,
            actions=False,
        )
        if single is None:
            continue
        if single.empty:
            continue
        single = single.reset_index()
        single["Ticker"] = ticker
        frames.append(single)

    if not frames:
        raise RuntimeError("No price data returned by yfinance.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["Date"] = pd.to_datetime(long_df["Date"])
    # yfinance may return placeholder rows for unavailable symbols; remove all-null price rows.
    long_df = long_df.dropna(
        subset=["Open", "High", "Low", "Close", "Adj Close"], how="all"
    )

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
    available_tickers = set(full_data["Ticker"].unique())
    missing_tickers = [t for t in tickers if t not in available_tickers and t != "SPY"]

    before_path = output_dir / "prices_before_2023-08-08.csv"
    after_path = output_dir / "prices_after_2023-08-08.csv"
    universe_path = output_dir / "ticker_universe.csv"

    before_data.to_csv(before_path, index=False)
    after_data.to_csv(after_path, index=False)
    pd.DataFrame(UNIVERSE).to_csv(universe_path, index=False)

    print("Downloaded price data successfully.")
    print(f"Universe size: {len(UNIVERSE)} tickers")
    print(f"Before-event rows: {len(before_data):,} -> {before_path}")
    print(f"After-event rows: {len(after_data):,} -> {after_path}")
    print(f"Ticker universe -> {universe_path}")
    if missing_tickers:
        print(
            f"\nWarning: Missing monthly price data for {len(missing_tickers)} tickers"
        )
        print("  " + ", ".join(missing_tickers))
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
