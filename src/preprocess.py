import polars as pl
import polars.selectors as cs

def process_data():
    input_path = "data/raw/train_ver2.csv"
    output_path = "data/processed/train_optimized.parquet"

    # Lazily scan the CSV, explicitly forcing the mixed-type column to be a String
    lf = pl.scan_csv(
        input_path,
        schema_overrides={"indrel_1mes": pl.String}
    )

    # Downcast numerical types to prevent 16GB RAM crash
    lf_optimized = lf.with_columns([
        cs.float().cast(pl.Float32),
        cs.integer().cast(pl.Int32)
    ])

    # Stream the processed data directly to a compressed Parquet file
    print("Processing and streaming to Parquet...")
    lf_optimized.sink_parquet(output_path)
    print("Data processing complete.")

if __name__ == "__main__":
    process_data()