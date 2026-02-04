import pandas as pd
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Replace with your downloaded file names
file_stock_1 = os.path.join(
    script_dir, "Galaxy Entertainment Group Stock Price History.csv"
)
file_stock_2 = os.path.join(script_dir, "Sands China Stock Price History.csv")
ticker_1 = "Galaxy Entertainment Group"  # 0027.HK
ticker_2 = "Sands China"  # 1928.HK


def analyze_stock(file_path, ticker_name):
    # 1. Read data (Investing.com CSV format with comma separator, date in "Date" column)
    # Note: Investing.com numbers sometimes contain ',' (e.g. 1,200.00), need to handle this
    df = pd.read_csv(file_path)

    # Clean data: convert 'Price' column to numeric, remove commas
    df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(float)

    # 2. Ensure sorted by date (oldest first, newest last) for return calculation
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=True)

    # 3. Calculate daily returns
    # Formula: (today's price - yesterday's price) / yesterday's price
    df["Return"] = df["Price"].pct_change()

    # Remove first row (first day has no previous price, return is NaN)
    df = df.dropna(subset=["Return"])

    # 4. Calculate statistics
    mean_ret = df["Return"].mean()
    var_ret = df["Return"].var()
    std_ret = df["Return"].std()
    percentile_5 = df["Return"].quantile(0.05)

    return {
        "Ticker": ticker_name,
        "Mean Return": mean_ret,
        "Variance": var_ret,
        "Standard Deviation": std_ret,
        "5th Percentile": percentile_5,
    }


# Run analysis
result1 = analyze_stock(file_stock_1, ticker_1)
result2 = analyze_stock(file_stock_2, ticker_2)
# Print results
print("--- Analysis Results ---")
for res in [result1, result2]:
    print(f"Stock: {res['Ticker']}")
    print(f"Mean return: {res['Mean Return']:.6f} (or {res['Mean Return']*100:.4f}%)")
    print(f"Variance: {res['Variance']:.8f}")
    print(f"Standard deviation: {res['Standard Deviation']:.6f}")
    print(f"5th percentile: {res['5th Percentile']:.6f}")
    print("-" * 20)
