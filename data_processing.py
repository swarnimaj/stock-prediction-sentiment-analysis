import pandas as pd
import numpy as np
import re
import string
import glob
import os
from datetime import datetime

# --- Configuration ---
# Define the paths to your data files/folders
# Make sure the 'archive/' folder is in the same directory as your script,
# or provide the full path.
TWEET_DATA_PATH = 'archive/tweet.csv'
STOCK_DATA_FOLDER = 'archive/Stocks/'

# Define the date range for analysis
START_DATE = '2015-01-01'
END_DATE = '2019-12-31'

print("--- Starting Data Preprocessing ---")

# --- Task: Load and Clean Tweet Data ---

print(f"\n1. Loading tweet data from: {TWEET_DATA_PATH}")
try:
    # Load the tweet dataset
    df_tweets = pd.read_csv(TWEET_DATA_PATH)
    print("   Tweet data loaded successfully.")
    print(f"   Initial shape: {df_tweets.shape}")
except FileNotFoundError:
    print(f"   Error: File not found at {TWEET_DATA_PATH}")
    print("   Please ensure the 'archive' folder and 'tweet.csv' file are correctly placed.")
    exit() # Exit if the main data file is missing

# --- Initial Inspection ---
print("\n2. Initial Tweet Data Inspection:")
# Display basic info (column names, non-null counts, data types)
print("   DataFrame Info:")
df_tweets.info()
# Display the first few rows
print("\n   DataFrame Head:")
print(df_tweets.head())
# Check for missing values in each column
print("\n   Missing values per column:")
print(df_tweets.isnull().sum())

# --- Data Cleaning ---
print("\n3. Cleaning Tweet Data:")

# 3.1 Handle Missing Tweet Bodies
# VADER analysis requires text, so rows with missing 'body' are not useful.
initial_rows = len(df_tweets)
df_tweets.dropna(subset=['body'], inplace=True)
rows_dropped = initial_rows - len(df_tweets)
if rows_dropped > 0:
    print(f"   Dropped {rows_dropped} rows with missing tweet bodies.")
else:
    print("   No rows dropped due to missing tweet bodies.")

# 3.2 Convert 'post_date' to Datetime
# The 'post_date' column appears to be in Unix timestamp format (seconds since epoch).
print("   Converting 'post_date' column to datetime objects...")
try:
    df_tweets['post_date'] = pd.to_datetime(df_tweets['post_date'], unit='s')
    # Extract the date part only, as we'll aggregate daily
    df_tweets['date'] = df_tweets['post_date'].dt.date
    df_tweets['date'] = pd.to_datetime(df_tweets['date']) # Convert date back to datetime object for filtering
    print("   'post_date' converted and 'date' column created.")
except Exception as e:
    print(f"   Error converting 'post_date': {e}. Please check the column format.")
    # Handle error or exit if date conversion is critical
    exit()

# 3.3 Define Text Cleaning Function
def clean_tweet_text(text):
    """
    Applies basic cleaning to tweet text:
    - Converts to lowercase
    - Removes URLs
    - Removes user mentions (@username)
    - Removes the '#' symbol from hashtags (keeps the text)
    - Removes punctuation (important: VADER uses some punctuation like '!')
      This basic version removes all punctuation; refinement might be needed.
    - Removes numbers
    - Removes extra whitespace
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
    text = re.sub(r'\@\w+', '', text) # Remove mentions
    text = re.sub(r'#', '', text) # Remove hashtag symbol but keep the text
    # Remove punctuation - VADER benefits from some punctuation (e.g., '!')
    # A more refined approach might selectively keep certain punctuation.
    # For simplicity here, we remove all standard punctuation.
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

print("   Applying text cleaning to the 'body' column...")
# Apply the cleaning function to the 'body' column to create 'cleaned_body'
df_tweets['cleaned_body'] = df_tweets['body'].apply(clean_tweet_text)
print("   'cleaned_body' column created.")

# 3.4 Filter Tweets by Date Range
print(f"   Filtering tweets between {START_DATE} and {END_DATE}...")
# Ensure 'date' column is in datetime format before filtering
df_tweets_filtered = df_tweets[(df_tweets['date'] >= START_DATE) & (df_tweets['date'] <= END_DATE)].copy()
filter_rows_dropped = len(df_tweets) - len(df_tweets_filtered)
print(f"   Filtered out {filter_rows_dropped} tweets outside the date range.")
print(f"   Shape after date filtering: {df_tweets_filtered.shape}")
if not df_tweets_filtered.empty:
    print(f"   Remaining date range: {df_tweets_filtered['date'].min()} to {df_tweets_filtered['date'].max()}")
else:
    print("   Warning: No tweets found within the specified date range.")

# --- Display Cleaned Data Sample ---
print("\n4. Sample of Cleaned and Filtered Tweet Data:")
print(df_tweets_filtered[['date', 'body', 'cleaned_body', 'retweet_num', 'like_num']].head())

print("\n--- Tweet Data Loading and Initial Cleaning Complete ---")

# The 'df_tweets_filtered' DataFrame is now ready for the next steps:
# - Extracting Company Identifiers (Cashtags)
# - Sentiment Analysis
# - Aggregation

# You can save this intermediate DataFrame if needed:
# df_tweets_filtered.to_csv('cleaned_filtered_tweets.csv', index=False)

def extract_ticker(text):
    """
    Extracts the first cashtag (e.g., $AAPL) from a text string using re.search.
    Returns the uppercase ticker symbol without the '$' or None if not found.
    """
    if not isinstance(text, str):
        return None
    # Search for the first occurrence of the pattern [1]
    match = re.search(r'\$[a-zA-Z]{1,5}\b', text)
    if match:
        # If a match is found, get the matched string, remove '$', and uppercase [1]
        return match.group()[1:].upper()
    # If no match is found, return None
    return None

# --- Task: Extract Tickers and Identify Top Companies ---

print("\n--- Extracting Tickers and Identifying Top Companies ---")

print("\n5. Extracting stock tickers (cashtags) from 'body' column...")
# Apply the function to the original 'body' column
df_tweets_filtered['ticker'] = df_tweets_filtered['body'].apply(extract_ticker)

# --- Inspect Ticker Extraction ---
print("   Sample of extracted tickers:")
print(df_tweets_filtered[['body', 'ticker']].head(10))

# 5.1 Handle Missing Tickers
initial_rows_filtered = len(df_tweets_filtered)
# Drop rows where no ticker was extracted
df_tweets_filtered.dropna(subset=['ticker'], inplace=True)
rows_dropped_no_ticker = initial_rows_filtered - len(df_tweets_filtered)
if rows_dropped_no_ticker > 0:
    print(f"   Dropped {rows_dropped_no_ticker} rows where no ticker could be extracted.")
else:
    print("   No rows dropped due to missing tickers.")
print(f"   Shape after dropping rows with no tickers: {df_tweets_filtered.shape}")

# 6. Identify Top 10 Companies by Tweet Volume
print("\n6. Identifying top 10 companies by tweet volume...")
# Count tweets per ticker
ticker_counts = df_tweets_filtered['ticker'].value_counts() # [1, 2, 3]

# Get the top 10 tickers
top_10_tickers = ticker_counts.head(10).index.tolist()

print("\n   Top 10 Tickers by Tweet Count:")
print(ticker_counts.head(10))
print(f"\n   Selected Top 10 Tickers: {top_10_tickers}")

# 7. Filter Tweets for Top 10 Companies
print("\n7. Filtering tweets to include only the top 10 companies...")
initial_rows_top10 = len(df_tweets_filtered)
# Keep only rows where the ticker is in the top 10 list
df_tweets_top10 = df_tweets_filtered[df_tweets_filtered['ticker'].isin(top_10_tickers)].copy()
rows_dropped_not_top10 = initial_rows_top10 - len(df_tweets_top10)
print(f"   Filtered out {rows_dropped_not_top10} tweets not related to the top 10 tickers.")
print(f"   Final shape of tweet data: {df_tweets_top10.shape}")

# --- Display Final Tweet Data Sample ---
print("\n8. Sample of Final Tweet Data (Top 10 Companies):")
print(df_tweets_top10[['date', 'ticker', 'cleaned_body']].head())

print("\n--- Ticker Extraction and Top 10 Company Selection Complete ---")

# The 'df_tweets_top10' DataFrame now contains cleaned tweets,
# filtered by date range and restricted to the 10 companies
# with the most tweets in the dataset during that period.
# It's ready for the next steps: Loading Stock Data.

# You can save this intermediate DataFrame if needed:
# df_tweets_top10.to_csv('cleaned_filtered_top10_tweets.csv', index=False)

import pandas as pd
import numpy as np
import glob # To find files matching a pattern
import os   # To handle file paths

# --- Configuration (assuming from previous steps) ---
STOCK_DATA_FOLDER = 'archive/Stocks/'
START_DATE = '2015-01-01'
END_DATE = '2019-12-31'
# Assuming 'top_10_tickers' list exists from the previous step
# Example: top_10_tickers =
# If not running sequentially, ensure top_10_tickers is defined here.
# Make sure tickers are uppercase and match the format derived from filenames (e.g., 'a.us.txt' -> 'A')

print("\n--- Loading and Preprocessing Stock Data ---")

# 8. Load Stock Data for Top 10 Companies
print(f"\n8. Loading stock data for top 10 tickers from: {STOCK_DATA_FOLDER}")
all_stock_files = glob.glob(os.path.join(STOCK_DATA_FOLDER, "*.us.txt")) # Find all.us.txt files [7, 8, 9, 10, 11]
stock_data_list = []

# Define expected columns and their types for consistency
stock_dtypes = {
    'Date': 'str', # Read as string first, then convert
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'float64', # Often integer, but float handles NaNs better initially
    'OpenInt': 'int64' # Though likely irrelevant later
}

required_stock_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

for file_path in all_stock_files:
    # Extract ticker from filename (e.g., 'aapl.us.txt' -> 'AAPL')
    filename = os.path.basename(file_path)
    ticker = filename.split('.')[0].upper()  # Get first part before dot and convert to uppercase

    if ticker in top_10_tickers:
        try:
            # Read the CSV, specifying dtypes
            df_stock_single = pd.read_csv(file_path, dtype=stock_dtypes)

            # Check if required columns exist
            if not all(col in df_stock_single.columns for col in required_stock_columns):
                print(f"   Warning: Skipping {filename}. Missing required columns.")
                continue

            # Add the ticker symbol as a column
            df_stock_single['ticker'] = ticker
            stock_data_list.append(df_stock_single)
            # print(f"   Loaded data for {ticker}") # Uncomment for verbose loading status
        except Exception as e:
            print(f"   Warning: Could not read file {filename}. Error: {e}")

if not stock_data_list:
    print("   Error: No stock data loaded for the top 10 tickers. Please check paths and filenames.")
    exit()

# Concatenate all individual stock DataFrames into one
df_stocks_all = pd.concat(stock_data_list, ignore_index=True) # [7, 8, 9, 10, 11, 12]
print(f"\n   Combined stock data shape for top 10 tickers: {df_stocks_all.shape}")

# 9. Clean Stock Data
print("\n9. Cleaning combined stock data:")

# 9.1 Convert 'Date' column to datetime objects
print("   Converting 'Date' column to datetime objects...")
try:
    df_stocks_all['Date'] = pd.to_datetime(df_stocks_all['Date']) # [13, 14]
    print("   'Date' column converted successfully.")
except Exception as e:
    print(f"   Error converting 'Date' column in stock data: {e}. Exiting.")
    exit()

# 9.2 Filter Stock Data by Date Range
print(f"   Filtering stock data between {START_DATE} and {END_DATE}...")
df_stocks_filtered = df_stocks_all[
    (df_stocks_all['Date'] >= START_DATE) & (df_stocks_all['Date'] <= END_DATE)
].copy() # [1, 2, 3, 4, 5, 6]
filter_rows_dropped_stock = len(df_stocks_all) - len(df_stocks_filtered)
print(f"   Filtered out {filter_rows_dropped_stock} stock records outside the date range.")
print(f"   Shape after date filtering: {df_stocks_filtered.shape}")

if df_stocks_filtered.empty:
    print("   Error: No stock data found within the specified date range for the top 10 tickers.")
    exit()

# 9.3 Sort Data by Ticker and Date (Important for time series operations like ffill)
print("   Sorting data by ticker and date...")
df_stocks_filtered.sort_values(by=['ticker', 'Date'], inplace=True)

# 9.4 Handle Missing Values (NaNs)
print("   Handling missing values (NaNs)...")
print("   Initial missing values per column:")
print(df_stocks_filtered[required_stock_columns + ['ticker']].isnull().sum())

# Forward fill missing price/volume data - common for financial time series
# Apply forward fill within each ticker group to prevent filling across different stocks
print("   Applying forward fill (ffill) within each ticker group...")
df_stocks_filtered[required_stock_columns] = df_stocks_filtered.groupby('ticker')[required_stock_columns].ffill() # [13, 15, 16, 17, 18, 19, 20, 21]

# Check if any NaNs remain at the beginning of any ticker's series
print("\n   Missing values after forward fill:")
print(df_stocks_filtered[required_stock_columns + ['ticker']].isnull().sum())

# Optional: Backward fill any remaining NaNs (usually only at the very start)
# df_stocks_filtered.fillna(method='bfill', inplace=True)
# Optional: Drop rows if NaNs still exist (e.g., if a stock has no data at the start of the period)
initial_rows_stock_filt = len(df_stocks_filtered)
df_stocks_filtered.dropna(subset=required_stock_columns, inplace=True)
rows_dropped_nan = initial_rows_stock_filt - len(df_stocks_filtered)
if rows_dropped_nan > 0:
    print(f"   Dropped {rows_dropped_nan} rows with remaining NaN values after fill attempts.")

# 9.5 Verify Data Types
print("\n   Verifying data types...")
print(df_stocks_filtered[required_stock_columns + ['ticker']].dtypes)
# Ensure numeric columns are indeed numeric (read_csv with dtype helps, but double-check)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    # Attempt conversion if not already float, handling potential errors
    if not pd.api.types.is_numeric_dtype(df_stocks_filtered[col]):
         try:
             df_stocks_filtered[col] = pd.to_numeric(df_stocks_filtered[col], errors='coerce')
             print(f"      Converted column '{col}' to numeric.")
         except Exception as e:
             print(f"      Warning: Could not convert column '{col}' to numeric. Error: {e}")

# Drop rows where essential numeric conversion might have failed (created NaNs)
df_stocks_filtered.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)


# --- Display Cleaned Stock Data Sample ---
print("\n10. Sample of Cleaned and Filtered Stock Data:")
print(df_stocks_filtered[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].head())
print(df_stocks_filtered[['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].tail())
print(f"\n   Final shape of stock data: {df_stocks_filtered.shape}")

print("\n--- Stock Data Loading and Preprocessing Complete ---")

# The 'df_stocks_filtered' DataFrame now contains cleaned stock data
# for the top 10 companies within the specified date range.
# It's ready for merging with the tweet data.

# You can save this intermediate DataFrame if needed:
# df_stocks_filtered.to_csv('cleaned_filtered_top10_stocks.csv', index=False)

import pandas as pd
import numpy as np
import re
import string
import glob
import os
from datetime import datetime

# --- Configuration (assuming from previous steps) ---
STOCK_DATA_FOLDER = 'archive/Stocks/'
START_DATE = '2015-01-01'
END_DATE = '2019-12-31'
OUTPUT_FILENAME = 'merged_stock_tweet_data.csv' # Define output filename

# --- Load Intermediate Data (assuming df_tweets_top10 and df_stocks_filtered exist) ---
# If not running sequentially, load the previously saved intermediate files or
# re-run the previous code blocks first.
# Example:
# df_tweets_top10 = pd.read_csv('cleaned_filtered_top10_tweets.csv', parse_dates=['date'])
# df_stocks_filtered = pd.read_csv('cleaned_filtered_top10_stocks.csv', parse_dates=['Date'])
# top_10_tickers = df_stocks_filtered['ticker'].unique().tolist() # Recreate if needed

# Ensure date columns are in the correct format if loading from CSV
# df_tweets_top10['date'] = pd.to_datetime(df_tweets_top10['date'])
# df_stocks_filtered['Date'] = pd.to_datetime(df_stocks_filtered['Date'])

print("\n--- Merging Tweet Data and Stock Data ---")

# 11. Prepare for Merge
# Ensure date columns have the same name for merging
# We'll use 'date' as the common column name.
# Rename the stock data's 'Date' column to 'date'.
df_stocks_filtered.rename(columns={'Date': 'date'}, inplace=True)

# Check data types before merging
print("\n   Data types before merge:")
print("   Tweet Data ('date', 'ticker'):", df_tweets_top10[['date', 'ticker']].dtypes)
print("   Stock Data ('date', 'ticker'):", df_stocks_filtered[['date', 'ticker']].dtypes)

# Ensure merge keys are compatible
# 'date' should be datetime64[ns] in both
# 'ticker' should be object (string) in both
if not pd.api.types.is_datetime64_any_dtype(df_tweets_top10['date']):
    df_tweets_top10['date'] = pd.to_datetime(df_tweets_top10['date'])
if not pd.api.types.is_datetime64_any_dtype(df_stocks_filtered['date']):
    df_stocks_filtered['date'] = pd.to_datetime(df_stocks_filtered['date'])

# 12. Perform the Merge
print("\n12. Merging stock data into tweet data...")
# We perform a 'left' merge, keeping all tweets from df_tweets_top10
# and adding the corresponding stock data for that ticker on that specific date.
# If a tweet's date/ticker doesn't match a stock record (e.g., weekend tweet),
# the stock columns will have NaN for that row.
df_merged = pd.merge(
    df_tweets_top10,
    df_stocks_filtered,
    on=['date', 'ticker'], # Merge on both date and ticker [1, 2, 3]
    how='left' # Keep all tweets, add stock data where available [1]
)

print(f"   Shape after merge: {df_merged.shape}")
print("   Columns after merge:", df_merged.columns.tolist())

# --- Handle Potential Merge Issues ---
# Check for rows where stock data might be missing (e.g., tweets on non-trading days)
missing_stock_data_count = df_merged['Close'].isnull().sum()
if missing_stock_data_count > 0:
    print(f"   Warning: {missing_stock_data_count} tweets did not have matching stock data (e.g., weekends/holidays). Stock columns contain NaN for these.")
    # Decide how to handle these later (e.g., drop them if stock data is essential for the features)
    # For now, we keep them, but they might be dropped during feature engineering if stock features are needed.

# 13. Drop Unnecessary Columns
print("\n13. Dropping unnecessary columns...")
columns_to_drop = ['OpenInt', 'post_date', 'body', 'comment_num', 'tweet_id', 'writer']
# Drop columns only if they exist to avoid errors
existing_columns_to_drop = [col for col in columns_to_drop if col in df_merged.columns]
df_final = df_merged.drop(columns=existing_columns_to_drop)

print(f"   Dropped columns: {existing_columns_to_drop}")
print(f"   Final columns: {df_final.columns.tolist()}")
print(f"   Final shape: {df_final.shape}")

# --- Final Inspection ---
print("\n14. Final Merged Data Inspection:")
print("   DataFrame Info:")
df_final.info()
print("\n   Sample Data (Head):")
print(df_final.head())
print("\n   Sample Data (Tail):")
print(df_final.tail())
print("\n   Missing values check:")
print(df_final.isnull().sum())

# 15. Save the Merged Data
print(f"\n15. Saving merged data to: {OUTPUT_FILENAME}")
try:
    df_final.to_csv(OUTPUT_FILENAME, index=False)
    print("   Merged data saved successfully.")
except Exception as e:
    print(f"   Error saving file: {e}")

print("\n--- Data Preprocessing and Merging Complete ---")
print(f"The final dataset is saved as '{OUTPUT_FILENAME}'.")
print("You can now proceed to the analysis phase using this file.")