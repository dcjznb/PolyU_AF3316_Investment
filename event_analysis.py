from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


EVENT_DATE = pd.Timestamp("2023-03-10")
WINDOW_DAYS = 2  # days before and after the event


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


def download_daily_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily prices for the event window."""
    # Get a wider range to ensure we capture enough trading days
    start_date = (EVENT_DATE - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = (EVENT_DATE + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
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
            start=start_date,
            end=end_date,
            interval="1d",
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


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns and market-adjusted returns."""
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Calculate daily return
    df["Daily_Return"] = df.groupby("Ticker")["Adj Close"].pct_change()

    # Get SPY return for each date
    spy_returns = (
        df[df["Ticker"] == "SPY"].set_index("Date")["Daily_Return"].rename("SPY_Return")
    )

    df = df.merge(spy_returns, left_on="Date", right_index=True, how="left")

    # Market-adjusted return = stock return - market return
    df["Market_Adjusted_Return"] = df["Daily_Return"] - df["SPY_Return"]

    # Days relative to event (0 = event date)
    df["Days_to_Event"] = (df["Date"] - EVENT_DATE).dt.days

    return df


def filter_event_window(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to event window: 2 trading days before, event day, and 2 trading days after."""
    # Get unique trading dates sorted
    all_dates = sorted(df["Date"].unique())

    # Find event date index
    event_dates = [d for d in all_dates if d.date() == EVENT_DATE.date()]
    if not event_dates:
        print(
            f"Warning: Event date {EVENT_DATE.date()} not found in data. Using nearest date."
        )
        nearest_date = min(all_dates, key=lambda x: abs((x - EVENT_DATE).days))
        event_idx = all_dates.index(nearest_date)
    else:
        event_idx = all_dates.index(event_dates[0])

    # Select 2 before, event, 2 after (up to 5 trading days)
    start_idx = max(0, event_idx - 2)
    end_idx = min(len(all_dates), event_idx + 3)

    selected_dates = all_dates[start_idx:end_idx]

    df = df[df["Date"].isin(selected_dates)].copy()
    return df.reset_index(drop=True)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "event_study_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [item["ticker"] for item in UNIVERSE]
    daily_data = download_daily_prices(tickers)
    daily_with_returns = calculate_returns(daily_data)
    event_window = filter_event_window(daily_with_returns)

    # Remove SPY from the main output (it's for reference/calculation only)
    event_window_no_spy = event_window[event_window["Ticker"] != "SPY"].copy()

    output_path = output_dir / "event_window_daily_2023-03-08_to_2023-03-12.csv"
    event_window_no_spy.to_csv(output_path, index=False)

    print("Daily event analysis data generated successfully.")
    print(f"\nOutput: {output_path}")
    print(f"Event date: {EVENT_DATE.date()}")
    print(
        f"Window: {EVENT_DATE - pd.Timedelta(days=WINDOW_DAYS)} to {EVENT_DATE + pd.Timedelta(days=WINDOW_DAYS)}"
    )
    print(f"\nTotal rows (excluding SPY): {len(event_window_no_spy)}")
    print(
        f"Date range in data: {event_window_no_spy['Date'].min().date()} to {event_window_no_spy['Date'].max().date()}"
    )
    print(f"\nColumns: {list(event_window_no_spy.columns)}")
    print("\nPreview (first 15 rows):")
    print(event_window_no_spy.head(15).to_string(index=False))

    # Summary statistics by group
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY GROUP")
    print("=" * 80)
    for group in ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]:
        group_data = event_window_no_spy[event_window_no_spy["Group"] == group]
        if not group_data.empty:
            avg_return = group_data["Daily_Return"].mean()
            avg_adj_return = group_data["Market_Adjusted_Return"].mean()
            print(
                f"\n{group:20s} | Avg Daily Return: {avg_return:8.4f} | Avg Market-Adj Return: {avg_adj_return:8.4f}"
            )


if __name__ == "__main__":
    main()
