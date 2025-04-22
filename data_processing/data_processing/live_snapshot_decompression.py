import os
import datetime
from datetime import date, timedelta
from typing import Optional, Union, Literal, List, Dict, Any
import io
import gzip
import msgpack # pip install msgpack

# Conditionally import zstandard
try:
    import zstandard as zstd # pip install zstandard
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

def retrieve_and_decompress_data(
    base_folder_path: str,
    symbol: str,
    category: str,
    start_date: Union[str, date],
    end_date: Union[str, date],
    compression_format: Literal["gzip", "zstandard", "none"] = "gzip"
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves, decompresses, and deserializes snapshot data for a given
    symbol, category, and date range.

    Assumes data is stored in the structure:
    {base_folder_path}/{category}/{symbol}/{YYYY-MM-DD}_snapshot.{extension}
    where each file contains one or more MessagePack-encoded objects,
    optionally compressed.

    Args:
        base_folder_path: The root directory where live snapshot data is stored
                          (e.g., "live_snapshots").
        symbol: The crypto symbol (e.g., "CAKEUSDT").
        category: The market category (e.g., "spot").
        start_date: The start date (inclusive), as a 'YYYY-MM-DD' string or
                      a datetime.date object.
        end_date: The end date (inclusive), as a 'YYYY-MM-DD' string or
                    a datetime.date object.
        compression_format: The compression used when saving ('gzip',
                             'zstandard', or 'none' if only msgpack was used).
                             Defaults to 'gzip'.

    Returns:
        A list of snapshot dictionaries found within the date range,
        or None if no data files are found or a critical error occurs
        (like missing compression library). Returns an empty list if
        files exist but contain no valid data.
    """
    symbol_path = os.path.join(base_folder_path, category, symbol)

    if not os.path.isdir(symbol_path):
        print(f"Error: Directory not found: {symbol_path}")
        return None

    # --- Determine file extension and check library availability ---
    if compression_format == "gzip":
        extension = "msgpack.gz"
    elif compression_format == "zstandard":
        if not ZSTD_AVAILABLE:
            print("Error: 'zstandard' compression format specified, but the "
                  "'zstandard' library is not installed. "
                  "Please run 'pip install zstandard'.")
            return None
        extension = "msgpack.zst"
    elif compression_format == "none":
        extension = "msgpack"
    else:
        print(f"Error: Unsupported compression_format: {compression_format}")
        return None

    # --- Validate and convert dates ---
    try:
        if isinstance(start_date, str):
            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        elif isinstance(start_date, date):
            start_dt = start_date
        else:
            raise TypeError("start_date must be a string or date object")

        if isinstance(end_date, str):
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        elif isinstance(end_date, date):
            end_dt = end_date
        else:
            raise TypeError("end_date must be a string or date object")

        if start_dt > end_dt:
            print(f"Error: Start date ({start_dt}) cannot be after end date ({end_dt})")
            return None
    except (ValueError, TypeError) as e:
        print(f"Error parsing dates: {e}")
        return None

    # --- Iterate through dates, decompress, and unpack ---
    all_snapshots: List[Dict[str, Any]] = []
    data_found_in_files = False # Track if any valid objects were unpacked
    files_processed_count = 0

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        file_name = f"{date_str}_snapshots.{extension}"
        file_path = os.path.join(symbol_path, file_name)

        if os.path.exists(file_path):
            files_processed_count += 1
            print(f"Processing file: {file_path}...")
            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    print(f"Info: Skipping empty file (0 bytes): {file_path}")
                    continue # Skip genuinely empty files

                # --- Open file with appropriate decompression ---
                stream = None
                if compression_format == "gzip":
                    # Gzip can raise EOFError if file is corrupt/empty header
                    try:
                        stream = gzip.open(file_path, "rb")
                    except EOFError:
                         print(f"Warning: Skipping potentially empty or corrupt gzip file (EOFError on open): {file_path}")
                         continue
                elif compression_format == "zstandard":
                    # Use streaming decompression for zstd
                    f_raw = open(file_path, 'rb')
                    dctx = zstd.ZstdDecompressor()
                    # stream_reader needs a context manager
                    # We'll manage it below along with f_raw
                    stream = dctx.stream_reader(f_raw)
                elif compression_format == "none":
                    stream = open(file_path, "rb")

                # --- Unpack data using msgpack.Unpacker ---
                if stream:
                    unpacker = msgpack.Unpacker(stream, raw=False, strict_map_key=False)
                    try:
                        unpacked_count_in_file = 0
                        for unpacked_item in unpacker:
                            all_snapshots.append(unpacked_item)
                            data_found_in_files = True
                            unpacked_count_in_file += 1
                        if unpacked_count_in_file > 0:
                             print(f"  -> Unpacked {unpacked_count_in_file} snapshot(s).")
                        else:
                             # Check if it was just an empty stream after decompression
                             print(f"Info: File {file_path} contained no MessagePack objects after decompression.")

                    except (msgpack.exceptions.UnpackException,
                            msgpack.exceptions.ExtraData, # Can occur if data is corrupt
                            ValueError, # Can be raised by unpacker on bad data
                            zstd.ZstdError if ZSTD_AVAILABLE else Exception, # Catch zstd errors during streaming read
                            EOFError # Can happen reading from compressed streams
                           ) as unpack_err:
                        print(f"Warning: Error unpacking data in {file_path} (size: {file_size}): {unpack_err}. Skipping rest of file.")
                    finally:
                        # Ensure streams/files are closed
                        if hasattr(stream, 'close'):
                            stream.close()
                        # For zstd, the raw file handle also needs closing if stream_reader didn't
                        if compression_format == "zstandard" and 'f_raw' in locals() and not f_raw.closed:
                             f_raw.close()


            except IOError as e:
                print(f"Warning: Could not read file {file_path}: {e}")
            except Exception as e:
                # Catch unexpected errors during file processing
                print(f"Warning: An unexpected error occurred processing {file_path}: {e}")
                # Ensure streams/files are closed even on unexpected errors
                if 'stream' in locals() and hasattr(stream, 'close') and not stream.closed:
                    stream.close()
                if 'f_raw' in locals() and compression_format == "zstandard" and not f_raw.closed:
                     f_raw.close()


        # else:
            # Optional: print(f"Info: File not found for date {date_str}: {file_path}")

        current_dt += timedelta(days=1) # Move to the next day

    if files_processed_count == 0:
        print(f"No data files found matching pattern for {symbol}/{category} between {start_dt} and {end_dt}")
        return None # Indicate no files matching criteria were found

    if not data_found_in_files and files_processed_count > 0:
        print(f"Processed {files_processed_count} file(s), but found no valid snapshot data.")
        # Return empty list, as files were processed but empty/invalid
        return []

    print(f"Successfully retrieved and unpacked {len(all_snapshots)} snapshots from {files_processed_count} file(s).")
    return all_snapshots

# --- Example Usage ---
if __name__ == "__main__":
    BASE_DATA_FOLDER = "live_snapshots" # Adjust to your actual folder
    SYMBOL_TO_GET = "CAKEUSDT"
    CATEGORY_TO_GET = "spot"
    START = "2025-04-22" # Example start date (use a date where you have data)
    END = "2025-04-22"   # Example end date
    # Choose the format your files were saved in:
    COMPRESSION = "gzip" # or "zstandard" or "none"

    print(f"Attempting to retrieve and decompress data for {SYMBOL_TO_GET}/{CATEGORY_TO_GET} from {START} to {END}...")

    snapshots = retrieve_and_decompress_data(
        BASE_DATA_FOLDER,
        SYMBOL_TO_GET,
        CATEGORY_TO_GET,
        START,
        END,
        compression_format=COMPRESSION
    )

    if snapshots is None:
        print("\nNo data files found or critical error occurred.")
    elif not snapshots: # Empty list
        print("\nFiles were found/processed, but they contained no valid snapshot data.")
    else:
        print(f"\nSuccessfully retrieved {len(snapshots)} snapshots.")
        print("Example snapshot (first one):")
        print(snapshots[:400])
        # You can now work with the list of dictionaries directly
        # e.g., load into pandas:
        # import pandas as pd
        # df = pd.DataFrame(snapshots)
        # print("\nDataFrame head:")
        # print(df.head())
