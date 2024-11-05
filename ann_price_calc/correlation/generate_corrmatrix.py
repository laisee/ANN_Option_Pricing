
import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_correlation_heatmap(correlation_matrix):
    """
    Create a correlation heatmap with compact formatting.
    
    Parameters:
    correlation_matrix (DataFrame): The correlation matrix to visualize
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with customized formatting
    sns.heatmap(correlation_matrix,
                annot=False,  # Show correlation values
                cmap='coolwarm',  # Color scheme
                vmin=-1, vmax=1,  # Correlation range
                center=0,  # Center the colormap at 0
                fmt='.2f',  # Round correlation values to 2 decimal places
                annot_kws={'size': 4},  # Smaller font for correlation values
                square=True)  # Make cells square
    
    # Customize axis labels and title
    plt.title('Asset Price Correlation Matrix', pad=10, fontsize=10)
    
    # Rotate x-axis labels and adjust their size
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

def analyze_symbol_correlations(db_path, table_name):
    """
    Extract price data from DuckDB and calculate correlations between symbols.
    
    Parameters:
    db_path (str): Path to the DuckDB database file
    table_name (str): Name of the table containing price data
    
    Returns:
    tuple: (DataFrame with pivoted data, DataFrame with correlations)
    """
    # Connect to DuckDB
    conn = duckdb.connect(db_path)
    
    try:
        # Extract data into pandas DataFrame
        query = f"""
        SELECT market_date, symbol, (open + high + low + close)/4 as "price"
        FROM {table_name}
        ORDER BY market_date, symbol
        """
        df = conn.execute(query).df()
        
        # Pivot the data to create columns for each symbol
        pivoted_df = df.pivot(index='market_date', columns='symbol', values='price')
        
        # Calculate correlation matrix between symbols
        correlation_matrix = pivoted_df.corr()
        
        # Create the heatmap
        create_correlation_heatmap(correlation_matrix)
        
        return pivoted_df, correlation_matrix
        
    finally:
        conn.close()

def get_highest_correlations(correlation_matrix, threshold=0.8):
    """
    Find pairs of highly correlated symbols.
    
    Parameters:
    correlation_matrix (DataFrame): The correlation matrix
    threshold (float): Minimum correlation value to include
    
    Returns:
    DataFrame: Pairs of highly correlated symbols
    """
    # Get upper triangle to avoid duplicates
    upper = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Stack the correlations and filter by threshold
    high_correlations = []
    for symbol1 in upper.columns:
        for symbol2 in upper.index:
            value = upper.loc[symbol2, symbol1]
            if value is not None and abs(value) >= threshold:
                high_correlations.append({
                    'Symbol1': symbol2,
                    'Symbol2': symbol1,
                    'Correlation': value
                })
    
    # Create DataFrame and sort by absolute correlation value
    corr_df = pd.DataFrame(high_correlations)
    if not corr_df.empty:
        corr_df = corr_df.sort_values(
            'Correlation', 
            key=abs,
            ascending=False
        )
    
    return corr_df

def print_correlation_summary(correlation_matrix):
    """
    Print a summary of correlation statistics for each symbol.
    
    Parameters:
    correlation_matrix (DataFrame): The correlation matrix
    """
    summary = []
    for symbol in correlation_matrix.columns:
        correlations = correlation_matrix[symbol].drop(symbol)
        summary.append({
            'Symbol': symbol,
            'Avg Correlation': correlations.mean(),
            'Max Correlation': correlations.max(),
            'Min Correlation': correlations.min(),
            'Most Correlated With': correlations.idxmax(),
            'Least Correlated With': correlations.idxmin()
        })
    
    summary_df = pd.DataFrame(summary)
    print("\nCorrelation Summary for each Symbol:")
    pd.set_option('display.max_rows', None)  # Show all rows
    print(summary_df.round(3))

# Example usage
if __name__ == "__main__":
    DB_PATH = "/Users/gavincostin/Downloads/binance_perps.db"
    TABLE_NAME = "market_data"
    
    # Get data and correlations
    price_data, correlations = analyze_symbol_correlations(DB_PATH, TABLE_NAME)
    
    # Find highly correlated pairs
    high_corr = get_highest_correlations(correlations, threshold=0.8)
    print("\nHighly correlated symbol pairs (correlation >= 0.8):")
    print(high_corr.round(3))
    
    # Print correlation summary
    print_correlation_summary(correlations)
    
    # Display the plot
    plt.show()
