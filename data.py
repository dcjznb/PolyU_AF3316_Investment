from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

START_DATE = "2014-03-01"
END_DATE = "2024-04-01"
EVENT_DATE = pd.Timestamp("2019-03-11")


UNIVERSE = [
    {"ticker": "THYAO.IS", "name": "Turkish Airlines", "group": "Treatment Big"},
    {"ticker": "RYAAY", "name": "Ryanair Holdings", "group": "Treatment Big"},
    {"ticker": "UAL", "name": "United Airlines Holdings", "group": "Treatment Big"},
    {
        "ticker": "ICAGY",
        "name": "International Consolidated Airlines Group ADR",
        "group": "Treatment Big",
    },
    {"ticker": "LUV", "name": "Southwest Airlines", "group": "Treatment Big"},
    {"ticker": "NAS.OL", "name": "Norwegian Air Shuttle", "group": "Treatment Big"},
    {"ticker": "AAL", "name": "American Airlines Group", "group": "Treatment Big"},
    {"ticker": "AC.TO", "name": "Air Canada", "group": "Treatment Big"},
    {"ticker": "CPA", "name": "Copa Holdings", "group": "Treatment Small"},
    {"ticker": "ALK", "name": "Alaska Air Group", "group": "Treatment Small"},
    {"ticker": "TUI1.DE", "name": "TUI Group", "group": "Treatment Small"},
    {"ticker": "JET2.L", "name": "Jet2 plc", "group": "Treatment Small"},
    {"ticker": "ALGT", "name": "Allegiant Travel", "group": "Treatment Small"},
    {"ticker": "AAVVF", "name": "Avianca Holdings SA", "group": "Treatment Small"},
    {"ticker": "CHR.TO", "name": "Chorus Aviation", "group": "Treatment Small"},
    {"ticker": "LTM.SN", "name": "LATAM Airlines Group", "group": "Control Big"},
    {"ticker": "9202.T", "name": "ANA Holdings", "group": "Control Big"},
    {"ticker": "9201.T", "name": "Japan Airlines", "group": "Control Big"},
    {"ticker": "0670.HK", "name": "China Eastern Airlines", "group": "Control Big"},
    {"ticker": "1055.HK", "name": "China Southern Airlines", "group": "Control Big"},
    {"ticker": "0293.HK", "name": "Cathay Pacific Airways", "group": "Control Big"},
    {"ticker": "DAL", "name": "Delta Air Lines", "group": "Control Big"},
    {"ticker": "C6L.SI", "name": "Singapore Airlines", "group": "Control Big"},
    {
        "ticker": "IAG.L",
        "name": "International Airlines Group",
        "group": "Control Small",
    },
    {"ticker": "QAN.AX", "name": "Qantas Airways", "group": "Control Small"},
    {"ticker": "LHA.DE", "name": "Lufthansa Group", "group": "Control Small"},
    {"ticker": "EZJ.L", "name": "easyJet", "group": "Control Small"},
    {"ticker": "AF.PA", "name": "Air France-KLM", "group": "Control Small"},
    {"ticker": "AIR.NZ", "name": "Air New Zealand", "group": "Control Small"},
    {"ticker": "WIZZ.L", "name": "Wizz Air Holdings", "group": "Control Small"},
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

    before_path = output_dir / "prices_before_2019-03-11.csv"
    after_path = output_dir / "prices_after_2019-03-11.csv"
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
