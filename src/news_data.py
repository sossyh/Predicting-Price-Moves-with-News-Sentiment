import pandas as pd

def load_news_data(filepath):
    # Load with explicit dtype for date column
    df = pd.read_csv(filepath, dtype={'date': str})
    
    # Debug: Show raw data sample
    print("\n=== RAW DATE SAMPLES ===")
    print(df['date'].head(10).to_string())
    
    # Custom parser with exhaustive format attempts
    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        
        # Try common patterns in order of likelihood
        formats = [
            '%Y-%m-%d %H:%M:%S%z',  # ISO with timezone
            '%Y-%m-%d %H:%M:%S',     # ISO without timezone
            '%m/%d/%Y %H:%M',       # US with time
            '%m/%d/%Y',             # US without time
            '%d-%m-%Y %H:%M',        # European with time
            '%d-%m-%Y',             # European without time
            '%Y-%m-%d'              # ISO date only
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt, exact=True)
            except:
                continue
        return pd.NaT
    
    # Apply parser
    df['date'] = df['date'].apply(parse_date)
    
    # Debug: Show parsing results
    print("\n=== PARSED DATE SAMPLES ===")
    print(df['date'].head(10).to_string())
    print(f"\nFailed to parse: {df['date'].isna().sum()} rows")
    
    # Final validation
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        raise ValueError("CRITICAL: Date parsing completely failed. First bad values:\n" + 
                       str(df[df['date'].isna()]['date'].head(5)))
    
    # Convert to date-only
    df['date'] = df['date'].dt.date
    
    # Clean data
    df = df.dropna(subset=['headline', 'date', 'stock'])
    df = df.drop_duplicates(subset=['headline', 'date', 'stock'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df