# analysis_improved.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json # To save metrics dict

# Import VADER (using NLTK)
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except LookupError:
    print("VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import Technical Analysis library
try:
    import ta
except ImportError:
    print("Technical Analysis library (ta) not found. Please install it: pip install ta")
    exit()

# Import scikit-learn components
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.feature_selection import VarianceThreshold

# Import XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost library not found. Please install it: pip install xgboost")
    exit()

# --- Configuration ---
INPUT_FILENAME = 'merged_stock_tweet_data.csv'
OUTPUT_DIR = 'analysis_results_improved' # Directory to save results
N_SPLITS_CV = 5 # Number of splits for TimeSeriesSplit (Outer loop)
N_SPLITS_TUNE = 3 # Number of splits for TimeSeriesSplit during tuning (Inner loop)
RANDOM_STATE = 42 # For reproducibility

# Feature Engineering Params
VOLATILITY_WINDOW = 21 # Approx 1 month trading days
RSI_WINDOW = 14
SMA_SHORT_WINDOW = 10
SMA_LONG_WINDOW = 30
FEATURE_LAGS = [1, 3, 5] # Lags in days to generate features for (e.g., 1, 3, 5 days ago)
TARGET_LAG = 1 # Predict next day's movement

# Model Tuning Params (Random Forest Example - Adjust grid as needed)
RF_PARAM_GRID = {
    'n_estimators': [100, 200], # Fewer estimators for quicker testing
    'max_depth': [10, None], # Test limited depth and unlimited
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}

# --- Helper Functions ---
def get_vader_compound(text, analyzer):
    """Calculates VADER compound score safely."""
    if not isinstance(text, str): return 0.0
    try: return analyzer.polarity_scores(text)['compound']
    except Exception: return 0.0

def evaluate_model_fold(y_true, y_pred):
    """Calculates classification metrics for a single fold."""
    accuracy = accuracy_score(y_true, y_pred)
    # Use pos_label=1 for 'Up' class metrics
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics = {'accuracy': accuracy, 'precision_1': precision, 'recall_1': recall, 'f1_1': f1}
    return metrics

def save_final_results(all_true, all_pred, model_name, metrics_list):
    """Calculates overall metrics, saves report, confusion matrix, and returns average metrics."""
    print(f"\n--- Overall Results for {model_name} ---")
    report = classification_report(all_true, all_pred, target_names=['Down (0)', 'Up (1)'], zero_division=0)
    print("Overall Classification Report:")
    print(report)

    # Calculate average metrics from folds
    avg_metrics = {
        f'avg_{key}': np.mean([m[key] for m in metrics_list])
        for key in metrics_list[0].keys() # Assumes all folds have the same keys
    }
    print("\nAverage Metrics Across Folds:")
    for key, value in avg_metrics.items():
        print(f"{key.replace('_1', ' (Class 1)')}: {value:.4f}")

    # Save Classification Report
    report_path = os.path.join(OUTPUT_DIR, f'classification_report_{model_name}.txt')
    with open(report_path, 'w') as f:
        f.write(f"--- {model_name} Overall Classification Report ---\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    # Plot and Save Confusion Matrix
    cm = confusion_matrix(all_true, all_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down (0)', 'Up (1)'])
    fig, ax = plt.subplots() # Create figure and axes explicitly
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f'Overall Confusion Matrix - {model_name} (All CV Folds)')
    cm_plot_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(cm_plot_path, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {cm_plot_path}")
    plt.close(fig) # Close the figure to prevent display / memory issues

    # Combine average metrics and the report string for saving
    full_metrics_data = {**avg_metrics, 'full_classification_report': report}
    return full_metrics_data

# --- Create Output Directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- 1. Load Preprocessed Data ---
print(f"--- 1. Loading Data from {INPUT_FILENAME} ---")
try:
    df = pd.read_csv(INPUT_FILENAME, parse_dates=['date'])
    print(f"   Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"   Error: File not found at {INPUT_FILENAME}. Run data_processing.py first.")
    exit()

# --- 2. Sentiment Analysis (VADER) ---
print("\n--- 2. Performing Sentiment Analysis with VADER ---")
sia = SentimentIntensityAnalyzer()
df['cleaned_body'] = df['cleaned_body'].fillna('').astype(str)
df['sentiment_compound'] = df['cleaned_body'].apply(lambda text: get_vader_compound(text, sia))
print("   VADER sentiment scores calculated.")

# --- 3. Aggregate Daily Sentiment ---
print("\n--- 3. Aggregating Daily Sentiment Scores ---")
daily_sentiment = df.groupby(['date', 'ticker']).agg(
    mean_sentiment_compound=('sentiment_compound', 'mean'),
    std_sentiment_compound=('sentiment_compound', 'std'),
    daily_tweet_volume=('cleaned_body', 'size')
).reset_index()
# Fix FutureWarning: Use direct assignment instead of inplace=True on a slice
daily_sentiment['std_sentiment_compound'] = daily_sentiment['std_sentiment_compound'].fillna(0)
print("   Daily sentiment aggregated.")

# --- 4. Prepare Daily Stock Data and Merge ---
print("\n--- 4. Preparing Daily Stock Data and Merging ---")
stock_cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
# Ensure we only work with trading days that had stock data originally
df_stock_daily = df.dropna(subset=['Close'])[stock_cols].drop_duplicates(subset=['date', 'ticker'])
# Merge sentiment data onto the trading days
df_daily = pd.merge(df_stock_daily, daily_sentiment, on=['date', 'ticker'], how='left')
# Fill sentiment NaNs (trading days with no tweets) with 0
sentiment_cols_to_fill = ['mean_sentiment_compound', 'std_sentiment_compound', 'daily_tweet_volume']
fill_values = {col: 0 for col in sentiment_cols_to_fill if col in df_daily.columns}
df_daily = df_daily.fillna(value=fill_values) # Use DataFrame.fillna with a dict

# Sort values for consistency
df_daily.sort_values(by=['ticker', 'date'], inplace=True)
df_daily.reset_index(drop=True, inplace=True)
print(f"   Daily stock data prepared and merged. Shape: {df_daily.shape}")
if df_daily.isnull().sum().any(): # More robust check for any NaNs
    print("   Warning: NaNs still present after merge/fill:")
    print(df_daily.isnull().sum()[df_daily.isnull().sum() > 0])

# --- 5. Feature Engineering ---
print("\n--- 5. Engineering Features ---")
df_eng = df_daily.copy() # Work on a copy to keep df_daily clean
df_eng = df_eng.set_index('date') # Set date index for TA lib and easier time ops

# Group by ticker for calculations to avoid data leakage across stocks
grouped = df_eng.groupby('ticker', group_keys=False)

# 5.1 Basic Price/Volume Features
print("   Calculating basic returns, volume change, volatility...")
# Use transform for calculations needing group context but aligned index
df_eng['daily_return'] = grouped['Close'].transform(lambda x: x.pct_change())
df_eng['volume_change'] = grouped['Volume'].transform(lambda x: x.pct_change())
# Calculate volatility based on daily returns within each group
df_eng['volatility'] = grouped['daily_return'].transform(lambda x: x.rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std()) * np.sqrt(252)

# 5.2 Technical Indicators using 'ta' library
print("   Calculating Technical Indicators (RSI, SMA)...")
base_features = ['daily_return', 'volume_change', 'volatility']
ta_features = []
try:
    # Use transform with lambda for TA functions applied per group
    df_eng['rsi'] = grouped['Close'].transform(lambda x: ta.momentum.rsi(x, window=RSI_WINDOW))
    df_eng['sma_short'] = grouped['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=SMA_SHORT_WINDOW))
    df_eng['sma_long'] = grouped['Close'].transform(lambda x: ta.trend.sma_indicator(x, window=SMA_LONG_WINDOW))
    ta_features = ['rsi', 'sma_short', 'sma_long']
    print("   Technical Indicators calculated.")
except Exception as e:
    print(f"   Error calculating technical indicators: {e}. Check 'ta' installation and data. Skipping TA features.")

# 5.3 Sentiment Features (names only)
sentiment_features = ['mean_sentiment_compound', 'std_sentiment_compound', 'daily_tweet_volume']

# 5.4 Create Target Variable
print("   Creating target variable...")
df_eng['next_day_close'] = grouped['Close'].shift(-TARGET_LAG)
df_eng['target'] = (df_eng['next_day_close'] > df_eng['Close']).astype(int)

# Define target variable name
target = 'target'

# 5.5 Create Lagged Features
print(f"   Creating lagged features for lags: {FEATURE_LAGS} day(s)...")
all_feature_names = base_features + ta_features + sentiment_features
lagged_feature_names = []

# Reset index to make grouped shift easier
df_eng = df_eng.reset_index()
grouped_for_lag = df_eng.groupby('ticker')

for lag in FEATURE_LAGS:
    for feature in all_feature_names:
        if feature in df_eng.columns:
            lagged_col_name = f'{feature}_lag{lag}'
            # Apply grouped shift
            df_eng[lagged_col_name] = grouped_for_lag[feature].shift(lag)
            lagged_feature_names.append(lagged_col_name)
        else:
            print(f"      Warning: Feature '{feature}' not found for lagging.")

# Set date index back after lagging is done
df_eng = df_eng.set_index('date')

# --- Handle NaNs created by calculations & lags ---
print("   Handling NaNs created during feature engineering...")
initial_rows = len(df_eng)
# Drop rows with NaN in target or any feature column used for modeling
df_final = df_eng.dropna(subset=[target] + lagged_feature_names)
rows_dropped = initial_rows - len(df_final)
print(f"   Dropped {rows_dropped} rows due to NaNs (target or lagged features).")
print(f"   Final shape for modeling: {df_final.shape}")

# Define final feature list
features = sorted(list(set(lagged_feature_names))) # Ensure unique features
target = 'target'

# --- Final Checks ---
if not features: print("Error: No lagged features created."); exit()
if df_final.empty: print("Error: DataFrame is empty after dropping NaNs."); exit()
if target not in df_final.columns: print(f"Error: Target '{target}' not found."); exit()
features = [f for f in features if f in df_final.columns] # Ensure features exist
if not features: print("Error: No feature columns exist after NaN drop."); exit()

print(f"\n   Features for modeling ({len(features)}): {features}")
print(f"   Target variable: {target}")
print("\n   Sample of final data for modeling:")
# Display head including ticker and target for context
print(df_final[['ticker'] + features + [target]].head())

# --- 6. Feature Selection ---
print("\n--- 6. Performing Feature Selection ---")
X = df_final[features]
y = df_final[target]

# 6.1 Variance Threshold
print("   Applying Variance Threshold (threshold=0.0)...")
try:
    selector_var = VarianceThreshold(threshold=0.0)
    selector_var.fit(X)
    features_retained_var = X.columns[selector_var.get_support()]
    X_var = X[features_retained_var]
    print(f"   Features remaining after Variance Threshold: {len(features_retained_var)}")
    if len(features_retained_var) < len(features):
         dropped_features = set(features) - set(features_retained_var)
         print(f"   Dropped features (zero variance): {list(dropped_features)}")
    features = features_retained_var.tolist() # Update feature list
except Exception as e:
    print(f"   Error during Variance Threshold: {e}. Skipping.")
    X_var = X

if X_var.empty: print("Error: No features remaining after Variance Threshold."); exit()

# 6.2 Correlation Analysis
print("\n   Analyzing Feature Correlation (threshold > 0.95)...")
correlation_matrix = X_var.corr().abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
# Find features with correlation > 0.95 to any *other* feature
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

if to_drop:
    print(f"   Highly correlated features identified: {to_drop}")
    X_corr = X_var.drop(columns=to_drop)
    print(f"   Dropped highly correlated features.")
    features = X_corr.columns.tolist() # Update feature list
else:
    print("   No features dropped due to high correlation.")
    X_corr = X_var

print(f"   Features remaining after Correlation Analysis: {len(features)}")

if X_corr.empty: print("Error: No features remaining after Correlation Analysis."); exit()

# Visualize final correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(X_corr.corr(), annot=False, cmap='coolwarm', fmt=".2f") # Annot=False for many features
plt.title('Feature Correlation Matrix (After Selection)')
plt.tight_layout()
correlation_plot_path = os.path.join(OUTPUT_DIR, 'correlation_matrix_final.png')
plt.savefig(correlation_plot_path, bbox_inches='tight')
print(f"   Final correlation matrix plot saved to: {correlation_plot_path}")
plt.close() # Close plot

X_model = X_corr
y_model = y

# --- 7. Model Training & Evaluation with TimeSeriesSplit ---
print(f"\n--- 7. Training Models ({N_SPLITS_CV}-Fold TimeSeriesSplit) ---")

tscv_outer = TimeSeriesSplit(n_splits=N_SPLITS_CV)

# Store metrics and predictions for each model across folds
model_results_list = {'LR':[], 'RF':[], 'XGB':[]}
all_predictions = {'LR':[], 'RF':[], 'XGB':[]}
all_true_labels = {'LR':[], 'RF':[], 'XGB':[]}
best_rf_params = None

fold = 0
for train_index, test_index in tscv_outer.split(X_model):
    fold += 1
    print(f"\n   --- Outer Fold {fold}/{N_SPLITS_CV} ---")
    X_train, X_test = X_model.iloc[train_index], X_model.iloc[test_index]
    y_train, y_test = y_model.iloc[train_index], y_model.iloc[test_index]
    print(f"      Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Scaling within the fold
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("      Features scaled.")

    # --- Logistic Regression ---
    print("      Training Logistic Regression...")
    model_lr = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000, solver='liblinear') # Use liblinear for smaller datasets
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    metrics_lr = evaluate_model_fold(y_test, y_pred_lr)
    model_results_list['LR'].append(metrics_lr)
    all_predictions['LR'].extend(y_pred_lr)
    all_true_labels['LR'].extend(y_test)
    print(f"      LR Fold {fold} Metrics: {metrics_lr}")

    # --- Random Forest (with Tuning) ---
    print("      Tuning and Training Random Forest...")
    tscv_inner = TimeSeriesSplit(n_splits=N_SPLITS_TUNE)
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
    # Reduce grid search verbosity unless debugging
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=RF_PARAM_GRID,
                                  cv=tscv_inner, scoring='f1', n_jobs=-1, verbose=0)
    grid_search_rf.fit(X_train_scaled, y_train)
    best_rf = grid_search_rf.best_estimator_
    print(f"      Best RF Params (Fold {fold}): {grid_search_rf.best_params_}")
    if fold == N_SPLITS_CV: best_rf_params = grid_search_rf.best_params_
    y_pred_rf = best_rf.predict(X_test_scaled)
    metrics_rf = evaluate_model_fold(y_test, y_pred_rf)
    model_results_list['RF'].append(metrics_rf)
    all_predictions['RF'].extend(y_pred_rf)
    all_true_labels['RF'].extend(y_test)
    print(f"      RF Fold {fold} Metrics: {metrics_rf}")


    # --- XGBoost ---
    print("      Training XGBoost...")
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    model_xgb = XGBClassifier(scale_pos_weight=scale_pos_weight,
                              random_state=RANDOM_STATE, eval_metric='logloss',
                              use_label_encoder=False, n_jobs=-1)
    model_xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = model_xgb.predict(X_test_scaled)
    metrics_xgb = evaluate_model_fold(y_test, y_pred_xgb)
    model_results_list['XGB'].append(metrics_xgb)
    all_predictions['XGB'].extend(y_pred_xgb)
    all_true_labels['XGB'].extend(y_test)
    print(f"      XGB Fold {fold} Metrics: {metrics_xgb}")

print("\n   Cross-Validation Finished.")

# --- 8. Final Evaluation & Saving ---
print("\n--- 8. Evaluating Overall Model Performance and Saving ---")
final_metrics_summary = {}
full_results_text = "Model Performance Summary\n============================\n"

# Calculate and save results for each model
model_names = ['LR', 'RF', 'XGB']  # Define the model names we've used
for name in model_names:
    metrics_data = save_final_results(
        all_true_labels[name],
        all_predictions[name],
        name,
        model_results_list[name]
    )
    final_metrics_summary[name] = metrics_data
    full_results_text += f"\n--- {name} ---" + \
                      f"\nAvg Accuracy:  {metrics_data['avg_accuracy']:.4f}" + \
                      f"\nAvg Precision: {metrics_data['avg_precision_1']:.4f}" + \
                      f"\nAvg Recall:    {metrics_data['avg_recall_1']:.4f}" + \
                      f"\nAvg F1:        {metrics_data['avg_f1_1']:.4f}" + \
                      "\n" + metrics_data['full_classification_report'] + \
                      "\n====================\n"

# Save average metrics summary
metrics_file_path = os.path.join(OUTPUT_DIR, 'final_metrics_summary.json')
with open(metrics_file_path, 'w') as f:
    # Keep only average values for JSON summary
    avg_only_metrics = {model: {k: v for k, v in data.items() if k.startswith('avg_')}
                        for model, data in final_metrics_summary.items()}
    json.dump(avg_only_metrics, f, indent=4)
print(f"\nFinal average metrics saved to: {metrics_file_path}")

# Save full text report
results_file_path = os.path.join(OUTPUT_DIR, 'full_results_summary.txt')
with open(results_file_path, 'w') as f:
    f.write(full_results_text)
print(f"Full results summary text file saved to: {results_file_path}")

# Save best RF parameters if found
if best_rf_params:
    rf_params_path = os.path.join(OUTPUT_DIR, 'best_randomforest_params.json')
    with open(rf_params_path, 'w') as f:
        json.dump(best_rf_params, f, indent=4)
    print(f"Best RandomForest parameters saved to: {rf_params_path}")

# --- 9. Feature Importance (using final models trained on full data) ---
print("\n--- 9. Calculating Feature Importance (Training on Full Data) ---")
# Use the final selected features (after correlation check)
X_full = X_model
y_full = y_model

# Scale the entire dataset
scaler_final = StandardScaler()
X_full_scaled = scaler_final.fit_transform(X_full)

# Random Forest Importance
if best_rf_params:
    print("   Calculating RF Feature Importance...")
    try:
        final_rf = RandomForestClassifier(**best_rf_params, random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1)
        final_rf.fit(X_full_scaled, y_full)
        rf_importances = final_rf.feature_importances_
        rf_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': rf_importances
        }).sort_values(by='Importance', ascending=False)

        print("\n   Random Forest Feature Importances (Top 20):")
        print(rf_importance_df.head(20))
        rf_importance_path = os.path.join(OUTPUT_DIR, 'feature_importances_randomforest.csv')
        rf_importance_df.to_csv(rf_importance_path, index=False)
        print(f"   RF Feature importance data saved to: {rf_importance_path}")

        # Plot RF importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=rf_importance_df.head(20))
        plt.title('Top 20 Feature Importance (Random Forest - MDI)')
        plt.tight_layout()
        rf_importance_plot_path = os.path.join(OUTPUT_DIR, 'feature_importance_plot_rf.png')
        plt.savefig(rf_importance_plot_path, bbox_inches='tight')
        print(f"   RF Feature importance plot saved to: {rf_importance_plot_path}")
        plt.close() # Close plot
    except Exception as e:
        print(f"   Error generating RF importance: {e}")
else:
    print("   Skipping RF Feature Importance (no best parameters found from tuning).")

# XGBoost Importance
print("\n   Calculating XGBoost Feature Importance...")
try:
    final_scale_pos_weight = sum(y_full == 0) / sum(y_full == 1) if sum(y_full == 1) > 0 else 1
    final_xgb = XGBClassifier(scale_pos_weight=final_scale_pos_weight,
                              random_state=RANDOM_STATE, eval_metric='logloss',
                              use_label_encoder=False, n_jobs=-1)
    final_xgb.fit(X_full_scaled, y_full) # Fit on full scaled data

    # Get importance using 'gain'
    xgb_importance_type = 'gain'
    xgb_importances_dict = final_xgb.get_booster().get_score(importance_type=xgb_importance_type)

    # Map importance scores back to original feature names if needed
    # Note: XGBoost importance dict might not include features with 0 importance
    xgb_importance_df = pd.DataFrame({
        'Feature': xgb_importances_dict.keys(),
        'Importance': xgb_importances_dict.values()
    }).sort_values(by='Importance', ascending=False)

    # Add features with zero importance if they are missing
    all_xgb_features = X_model.columns # Get all feature names input to XGB
    zero_importance_features = set(all_xgb_features) - set(xgb_importance_df['Feature'])
    if zero_importance_features:
        zero_df = pd.DataFrame({'Feature': list(zero_importance_features), 'Importance': 0.0})
        xgb_importance_df = pd.concat([xgb_importance_df, zero_df], ignore_index=True).sort_values(by='Importance', ascending=False)


    print(f"\n   XGBoost Feature Importances (Type: {xgb_importance_type}, Top 20):")
    print(xgb_importance_df.head(20))
    xgb_importance_path = os.path.join(OUTPUT_DIR, f'feature_importances_xgboost_{xgb_importance_type}.csv')
    xgb_importance_df.to_csv(xgb_importance_path, index=False)
    print(f"   XGB Feature importance data saved to: {xgb_importance_path}")

    # Plot XGB importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=xgb_importance_df.head(20))
    plt.title(f'Top 20 Feature Importance (XGBoost - {xgb_importance_type.capitalize()})')
    plt.tight_layout()
    xgb_importance_plot_path = os.path.join(OUTPUT_DIR, f'feature_importance_plot_xgb_{xgb_importance_type}.png')
    plt.savefig(xgb_importance_plot_path, bbox_inches='tight')
    print(f"   XGB Feature importance plot saved to: {xgb_importance_plot_path}")
    plt.close() # Close plot
except Exception as e:
    print(f"   Could not calculate or plot XGBoost feature importance: {e}")


# --- 10. Visualization: Stock Price vs Sentiment (Example) ---
print("\n--- 10. Visualizing Stock Price vs. Sentiment (Example Ticker) ---")
if not df_final.empty:
    available_tickers = df_final['ticker'].unique()
    if available_tickers.size > 0:
        # Select a sample ticker (e.g., the first one alphabetically)
        sample_ticker = sorted(available_tickers)[0]
        print(f"   Plotting for ticker: {sample_ticker}")
        # Use df_eng which has the non-lagged sentiment for easier plotting alignment
        df_sample = df_eng[df_eng['ticker'] == sample_ticker].copy() # Use df_eng before dropna

        # Check if required columns exist
        sentiment_col = 'mean_sentiment_compound' # Plot non-lagged for visual correlation
        if not df_sample.empty and sentiment_col in df_sample.columns and 'Close' in df_sample.columns:
            fig, ax1 = plt.subplots(figsize=(14, 7))
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Close Price', color=color)
            ax1.plot(df_sample.index, df_sample['Close'], color=color, label='Close Price') # Index is date here
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.tick_params(axis='x', rotation=45) # Rotate date labels

            ax2 = ax1.twinx() # Create secondary axis
            color = 'tab:red'
            ax2.set_ylabel(f'{sentiment_col}', color=color)
            ax2.plot(df_sample.index, df_sample[sentiment_col], color=color, linestyle='--', alpha=0.7, label=f'{sentiment_col}')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(0, color='grey', linestyle=':', linewidth=0.5) # Neutral sentiment line

            plt.title(f'{sample_ticker} Stock Price vs. Daily Mean Sentiment')
            fig.tight_layout()
            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')

            price_sentiment_plot_path = os.path.join(OUTPUT_DIR, f'{sample_ticker}_price_vs_sentiment.png')
            plt.savefig(price_sentiment_plot_path, bbox_inches='tight')
            print(f"   Price vs Sentiment plot saved to: {price_sentiment_plot_path}")
            plt.show() # Display the plot
            plt.close(fig) # Ensure plot is closed
        else:
            print(f"   Plotting failed: No data or required columns ('Close', '{sentiment_col}') found for ticker {sample_ticker}.")
    else:
         print("   Plotting failed: No tickers found in the final processed data.")
else:
    print("   Plotting failed: Final DataFrame is empty.")

print("\n--- Improved Analysis Script Finished ---")
print(f"All results and plots saved in the '{OUTPUT_DIR}' directory.")