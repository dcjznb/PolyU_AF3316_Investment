from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


EVENT_DATE = pd.Timestamp("2019-03-11")
WINDOW_DAYS = 2  # days before and after the event


UNIVERSE = [
    {"ticker": "BA", "name": "Boeing", "group": "Treatment Big"},
    {"ticker": "GE", "name": "General Electric", "group": "Treatment Big"},
    {"ticker": "LUV", "name": "Southwest Airlines", "group": "Treatment Big"},
    {"ticker": "AAL", "name": "American Airlines", "group": "Treatment Big"},
    {"ticker": "UAL", "name": "United Airlines", "group": "Treatment Big"},
    {"ticker": "TDG", "name": "TransDigm Group", "group": "Treatment Big"},
    {"ticker": "HEI", "name": "HEICO", "group": "Treatment Big"},
    {"ticker": "RYAAY", "name": "Ryanair Holdings ADR", "group": "Treatment Big"},
    {"ticker": "ATRO", "name": "Astronics", "group": "Treatment Small"},
    {"ticker": "ALK", "name": "Alaska Air Group", "group": "Treatment Small"},
    {"ticker": "HXL", "name": "Hexcel", "group": "Treatment Small"},
    {"ticker": "WWD", "name": "Woodward", "group": "Treatment Small"},
    {"ticker": "ATI", "name": "ATI", "group": "Treatment Small"},
    {"ticker": "MOG-A", "name": "Moog Class A", "group": "Treatment Small"},
    {"ticker": "AER", "name": "AerCap", "group": "Treatment Small"},
    {"ticker": "DAL", "name": "Delta Air Lines", "group": "Control Big"},
    {"ticker": "LMT", "name": "Lockheed Martin", "group": "Control Big"},
    {"ticker": "NOC", "name": "Northrop Grumman", "group": "Control Big"},
    {"ticker": "GD", "name": "General Dynamics", "group": "Control Big"},
    {"ticker": "HON", "name": "Honeywell", "group": "Control Big"},
    {"ticker": "FDX", "name": "FedEx", "group": "Control Big"},
    {"ticker": "UPS", "name": "United Parcel Service", "group": "Control Big"},
    {"ticker": "CSX", "name": "CSX", "group": "Control Big"},
    {"ticker": "JBLU", "name": "JetBlue Airways", "group": "Control Small"},
    {
        "ticker": "VLRS",
        "name": "Controladora Vuela ADR",
        "group": "Control Small",
    },
    {"ticker": "SKYW", "name": "SkyWest", "group": "Control Small"},
    {"ticker": "ALGT", "name": "Allegiant Travel", "group": "Control Small"},
    {
        "ticker": "HII",
        "name": "Huntington Ingalls Industries",
        "group": "Control Small",
    },
    {"ticker": "TXT", "name": "Textron", "group": "Control Small"},
    {"ticker": "LHX", "name": "L3Harris Technologies", "group": "Control Small"},
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


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns and market-adjusted returns."""
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Calculate daily return
    df["Daily_Return"] = df.groupby("Ticker")["Adj Close"].pct_change(fill_method=None)

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
    trading_day_map = {
        d: i - event_idx for i, d in enumerate(selected_dates, start=start_idx)
    }

    df = df[df["Date"].isin(selected_dates)].copy()
    df["Days_to_Event"] = df["Date"].map(trading_day_map)
    return df.reset_index(drop=True)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "event_study_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = [item["ticker"] for item in UNIVERSE]
    daily_data = download_daily_prices(tickers)
    available_tickers = set(daily_data["Ticker"].unique())
    missing_tickers = [t for t in tickers if t not in available_tickers and t != "SPY"]
    daily_with_returns = calculate_returns(daily_data)
    event_window = filter_event_window(daily_with_returns)

    # Remove SPY from the main output (it's for reference/calculation only)
    event_window_no_spy = event_window[event_window["Ticker"] != "SPY"].copy()

    output_path = output_dir / "event_window_daily_2019-03-07_to_2019-03-13.csv"
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
    if missing_tickers:
        print(
            f"\nWarning: Missing daily event-window data for {len(missing_tickers)} tickers"
        )
        print("  " + ", ".join(missing_tickers))
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
