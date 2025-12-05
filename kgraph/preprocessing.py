import pandas as pd
import ast
import re
import numpy as np
import string
import csv
import os
from urllib.parse import quote


def replace_urls_with_titles(df, column_name, url_to_title_mapping):
    """Replace URLs in a column with their corresponding titles from the mapping"""
    if column_name not in df.columns:
        return df

    def map_urls(entry):
        # Handle NaN values safely
        if isinstance(entry, (float, int)) and np.isnan(entry):
            return np.nan

        if entry is None:
            return np.nan

        # Handle list/array-like objects
        if isinstance(entry, list) or isinstance(entry, np.ndarray):
            mapped_list = []
            for item in entry:
                if item in url_to_title_mapping:
                    mapped_list.append(url_to_title_mapping[item])
                else:
                    mapped_list.append(item)
            return mapped_list if mapped_list else np.nan

        # Handle string
        elif isinstance(entry, str):
            # Try direct mapping
            if entry in url_to_title_mapping:
                return url_to_title_mapping[entry]

            # Try parsing as list
            try:
                if entry.startswith("[") and entry.endswith("]"):
                    urls = ast.literal_eval(entry)
                    if isinstance(urls, list):
                        mapped_list = []
                        for url in urls:
                            if url in url_to_title_mapping:
                                mapped_list.append(url_to_title_mapping[url])
                            else:
                                mapped_list.append(url)
                        return mapped_list if mapped_list else entry
            except:
                pass

            return entry

        # Return original for other types
        return entry

    try:
        df[column_name] = df[column_name].apply(map_urls)
    except Exception as e:
        print(f"Warning: Error replacing URLs in column {column_name}: {e}")
        # Try a more conservative approach if the first one fails
        try:
            df[column_name] = df[column_name].apply(
                lambda x: url_to_title_mapping.get(x, x) if isinstance(x, str) and x in url_to_title_mapping else x
            )
        except:
            pass

    return df


def fix_malformed_urls(entry):
    """Fix malformed URLs in entries"""
    # Handle NaN values safely
    if isinstance(entry, (float, int)) and np.isnan(entry):
        return entry

    if entry is None:
        return entry

    if isinstance(entry, str):
        try:
            # Handle both single URLs and lists of URLs
            if entry.startswith("[") and entry.endswith("]"):
                urls = ast.literal_eval(entry)
                if isinstance(urls, list):
                    return [re.sub(r"^https//", "https://", url.strip()) for url in urls if isinstance(url, str)]
            # Handle single URL case
            return re.sub(r"^https//", "https://", entry.strip())
        except:
            return entry

    # Handle list directly
    if isinstance(entry, list):
        return [re.sub(r"^https//", "https://", url.strip()) if isinstance(url, str) else url
                for url in entry]

    return entry


def clean_data(df):
    """Clean and preprocess the data"""
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Handle missing values
    df.replace(r"(?i)^not found$", np.nan, regex=True, inplace=True)
    df.replace("", np.nan, inplace=True)

    # Parse lists
    for col in df.columns:
        def parse_lists(value):
            # Handle NaN values safely
            if isinstance(value, (float, int)) and np.isnan(value):
                return np.nan

            if value is None:
                return np.nan

            # Parse list strings
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        return [item for item in parsed if not (isinstance(item, str) and
                                                                item.strip().lower() in ["nan", "null", "none"])]
                    return parsed
                except:
                    return value
            return value

        try:
            df[col] = df[col].apply(parse_lists)
        except Exception as e:
            print(f"Warning: Could not parse lists in column {col}: {e}")

    # Clean text
    punct_chars = set(string.punctuation)

    # Remove illegal characters from strings only
    for col in df.columns:
        try:
            df[col] = df[col].apply(
                lambda x: re.sub(r"[\000-\010]|[\013-\014]|[\016-\037]", "", str(x))
                if isinstance(x, str) else x
            )
        except Exception as e:
            print(f"Warning: Could not clean text in column {col}: {e}")

    # Clean specific columns that need to be URI-safe
    if 'Title' in df.columns:
        df['Title'] = df['Title'].apply(
            lambda x: "".join(char for char in str(x) if char not in punct_chars)
            if isinstance(x, str) else x
        )

    if 'Part of a series on' in df.columns:
        df['Part of a series on'] = df['Part of a series on'].apply(
            lambda x: "".join(char for char in str(x) if char not in punct_chars)
            if isinstance(x, str) else x
        )

    return df


def create_uri_safe_name(text):
    """Convert text to URI-safe format"""
    # Handle NaN values safely
    if isinstance(text, (float, int)) and np.isnan(text):
        return None

    if text is None or text == 'nan' or text == '':
        return None

    try:
        # Remove special characters and replace spaces with underscores
        text = re.sub(r'[^\w\s]', '', str(text))
        text = re.sub(r'\s+', '_', text.strip())

        # Ensure valid URI format
        return quote(text)
    except:
        return None


def read_and_clean_csv(file_path):
    """
    Read a CSV file and filter rows by valid Status values
    """
    try:
        # Standard reading - explicitly don't use the first column as index
        df = pd.read_csv(file_path, index_col=None)

        COLUMN_MAPPING = {
            # Core identification
            'url': 'KYM URL',
            'id': 'ID',
            'title': 'Title',

            # Status and main classification
            'status': 'Status',
            'entry_category': 'Entry Type',  # The main type: meme/person/event

            # Meme format classifications (the new distinction)
            'entry_type_names': 'Meme Format',  # Image Macro, Exploitable, etc.
            'entry_types': 'Meme Format Details',  # Full dict with URLs
            'entry_type_ids': 'Meme Format IDs',  # Just the IDs

            # Other classifications
            'type': 'Type:',  # Contains things like "character, exploitable, image macro"
            'badges': 'Badges:',

            # Temporal
            'year': 'Year',

            # Location/Origin
            'origin_location': 'Origin',
            'region': 'Region',

            # Series/Relations
            'series_name': 'Part of a series on',
            'series_link': 'Series Link',
            'related_entities_url': 'Related Entities',

            # Media/Content
            'recent_images': 'Recent Images URLs',
            'main_image_url': 'Main Image URL',
            'html_file': 'HTML File',

            # Engagement metrics
            'views': 'Views',
            'video_count': 'Video Count',
            'photo_count': 'Photo Count',
            'comment_count': 'Comment Count',

            # Text content
            'about_text': 'About Text',
            'origin_text': 'Origin Text',
            'spread_text': 'Spread Text',
            'full_text': 'Full Text',
            'meta_description': 'Meta Description',

            # Metadata
            'tags': 'Tags',
            'external_references': 'External References',
        }
        df.rename(columns=COLUMN_MAPPING, inplace=True)

        # Remove any unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed: ' in col]
        if unnamed_cols:
            print(f"Removing {len(unnamed_cols)} unnamed columns")
            df = df.drop(columns=unnamed_cols)

        original_row_count = len(df)

        # Filter to keep only rows with valid Status values
        if 'Status' in df.columns:
            # List of valid status values (case-sensitive since the file should have proper capitalization)
            valid_statuses = ['Confirmed', 'Unconfirmed', 'confirmed', 'unconfirmed', 'Submission']

            # Create a mask for rows with valid and non-empty status
            # First check for empty values
            empty_status_mask = df['Status'].isna() | (df['Status'].astype(str) == '') | (
                        df['Status'].astype(str) == 'nan')
            if empty_status_mask.any():
                print(f"Found {empty_status_mask.sum()} rows with empty Status")

            # Then check for invalid values
            invalid_status_mask = ~empty_status_mask & ~df['Status'].isin(valid_statuses)
            if invalid_status_mask.any():
                print(f"Found {invalid_status_mask.sum()} rows with invalid Status values")
                print("Invalid values found:", df.loc[invalid_status_mask, 'Status'].unique())

            # Combined mask for valid rows
            valid_mask = df['Status'].isin(valid_statuses)

            # Count total invalid rows
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                print(f"Total of {invalid_count} rows with invalid or missing Status will be removed")
                # Apply the filter
                df = df[valid_mask].reset_index(drop=True)
                print(f"Kept {len(df)} rows out of {original_row_count}")
            else:
                print("All rows have valid Status values")
        else:
            print("Warning: No 'Status' column found in file!")

        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")

        try:
            # Fallback with error handling - again, don't use first column as index
            df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True, index_col=None)

            # Remove any unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed: ' in col]
            if unnamed_cols:
                print(f"Removing {len(unnamed_cols)} unnamed columns")
                df = df.drop(columns=unnamed_cols)

            print(f"Loaded file with {len(df)} rows after skipping bad lines")

            # Apply the same status filtering
            if 'Status' in df.columns:
                valid_statuses = ['Confirmed', 'Submission']
                valid_mask = df['Status'].isin(valid_statuses)
                invalid_count = (~valid_mask).sum()
                if invalid_count > 0:
                    print(f"Removing {invalid_count} rows with invalid or missing Status")
                    df = df[valid_mask].reset_index(drop=True)

            return df
        except Exception as e2:
            print(f"All reading methods failed: {e2}")
            return None


def check_csv_overflow(file_path):
    """
    Check if a CSV file has overflow issues by reading it with different methods
    and comparing column counts and row lengths
    """
    print(f"\n=== Detailed CSV Overflow Check for {file_path} ===")

    try:
        # Method 1: Read with pandas standard reader - explicitly don't use first column as index
        df_pandas = pd.read_csv(file_path, index_col=None)
        pandas_cols = len(df_pandas.columns)
        pandas_rows = len(df_pandas)
        print(f"Pandas reader: {pandas_rows} rows, {pandas_cols} columns")

        # Method 2: Read with csv module line by line
        rows = []
        max_fields = 0
        min_fields = float('inf')
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            header_len = len(header)
            print(f"CSV header has {header_len} fields")

            # Check each row
            for i, row in enumerate(reader):
                row_len = len(row)
                max_fields = max(max_fields, row_len)
                min_fields = min(min_fields, row_len)

                # If this row has more fields than the header, it might be an overflow
                if row_len > header_len:
                    print(f"OVERFLOW: Row {i + 2} has {row_len} fields (header has {header_len})")
                    # Print a sample of the offending row
                    print(f"  First few fields: {row[:5]}")
                    print(f"  Last few fields: {row[-5:] if len(row) >= 5 else row}")

                rows.append(row)

        csv_rows = len(rows)
        print(f"CSV reader: {csv_rows} rows, min fields: {min_fields}, max fields: {max_fields}, header: {header_len}")

        # Check for inconsistencies
        if max_fields > header_len:
            print(f"WARNING: Some rows have more fields ({max_fields}) than the header ({header_len})")
            print("This will cause overflow when opened in Excel or other spreadsheet programs")

        if min_fields < header_len:
            print(f"WARNING: Some rows have fewer fields ({min_fields}) than the header ({header_len})")
            print("This might cause data alignment issues")

        # Check if rows with too many fields have quotes that should be escaped
        problem_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Count unescaped quotes
                quote_count = line.count('"') - line.count('\\"')
                if quote_count % 2 != 0:
                    problem_lines.append((i + 1, line[:100] + "..." if len(line) > 100 else line))

        if problem_lines:
            print(f"Found {len(problem_lines)} lines with unbalanced quotes that could cause parsing issues:")
            for line_num, line_text in problem_lines[:5]:  # Show first 5 examples
                print(f"  Line {line_num}: {line_text.strip()}")

        return pandas_rows == csv_rows and max_fields <= header_len

    except Exception as e:
        print(f"Error checking CSV overflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def validate_cleaned_data(file_path):
    """
    Validate that a cleaned CSV file contains only valid Status values
    """
    print(f"\nValidating cleaned file: {file_path}")
    try:
        # Load the cleaned file - explicitly don't use first column as index
        df = pd.read_csv(file_path, index_col=None)

        # Remove any unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed: ' in col]
        if unnamed_cols:
            print(f"Found {len(unnamed_cols)} unnamed columns that should be removed")

        print(f"File contains {len(df)} rows")

        # Check Status column
        if 'Status' in df.columns:
            # Count unique Status values
            status_counts = df['Status'].value_counts(dropna=False)
            print("Status value counts:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

            # Check for empty or invalid values
            valid_statuses = ['Confirmed', 'Unconfirmed', 'confirmed', 'unconfirmed', 'Submission']
            invalid_mask = ~df['Status'].isin(valid_statuses)
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                print(f"WARNING: Found {invalid_count} rows with invalid Status values!")
                # Print a sample of invalid rows
                print("Sample of invalid rows:")
                sample_invalid = df[invalid_mask].head(5)
                for idx, row in sample_invalid.iterrows():
                    print(f"  Row {idx}: Status = '{row['Status']}', Title = '{row.get('Title', 'N/A')}'")
            else:
                print("All Status values are valid (Confirmed or Submission)")

            # Check for specific string 'nan' as Status value (which can happen with data export/import)
            nan_string_mask = df['Status'].astype(str).isin(['nan', 'None', ''])
            nan_string_count = nan_string_mask.sum()
            if nan_string_count > 0:
                print(f"WARNING: Found {nan_string_count} rows with 'nan', 'None', or empty string as Status!")
        else:
            print("WARNING: No 'Status' column found in file!")

        return df
    except Exception as e:
        print(f"Error validating file: {str(e)}")
        return None


def process_meme_file(file_path, index_key_file_path=None, output_suffix="_clean.csv"):
    """Process a single meme file"""
    print(f"Processing {file_path}...")

    # Load data and filter by status
    try:
        data = read_and_clean_csv(file_path)
        if data is None:
            print(f"Could not load {file_path}")
            return None
        print(f"Loaded {len(data)} rows from {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Create URL to title mapping if index key file is provided
    url_to_title_mapping = {}
    if index_key_file_path and os.path.exists(index_key_file_path):
        try:
            index_key = pd.read_csv(index_key_file_path)

            # Fix URLs in index key
            index_key['URL'] = index_key['URL'].apply(fix_malformed_urls)
            if 'Title' in index_key.columns:
                index_key['Title'] = index_key['Title'].str.strip()

                # Determine the column name for KYM URL
                kym_url_col = None
                for col in ['KYM URL', 'URL', 'Link']:
                    if col in index_key.columns:
                        kym_url_col = col
                        break

                if kym_url_col:
                    # Create mapping dictionary
                    url_to_title_mapping = dict(zip(index_key[kym_url_col], index_key['Title']))
        except Exception as e:
            print(f"Warning: Could not process index key file: {e}")

    # Fix malformed URLs
    url_columns = ['Series Link', 'Related Entities', 'Recent Images URLs', 'Related Memes']
    for column in url_columns:
        if column in data.columns:
            try:
                data[column] = data[column].apply(fix_malformed_urls)
            except Exception as e:
                print(f"Warning: Could not fix URLs in column {column}: {e}")

    # Replace URLs with titles if mapping exists
    if url_to_title_mapping:
        target_columns = ['Related Memes', 'Related Entities', 'Series Link']
        for column in target_columns:
            if column in data.columns:
                try:
                    data = replace_urls_with_titles(data, column, url_to_title_mapping)
                except Exception as e:
                    print(f"Warning: Could not replace URLs in column {column}: {e}")

    # Clean data
    data = clean_data(data)

    # Create URI-safe columns and other new columns
    new_columns = {}

    # Add source filename as a column
    base_filename = os.path.basename(file_path)
    new_columns['Source_File'] = [base_filename] * len(data)

    # Process Title and Part of a series on for URI-safe versions
    uri_columns = ['Title', 'Part of a series on']
    for col in uri_columns:
        if col in data.columns:
            try:
                # Ensure strings are trimmed
                data[col] = data[col].apply(lambda x: str(x).strip() if not pd.isna(x) else x)

                # Create URI-safe version
                uri_col_name = f'URI_Safe_{col}'
                new_columns[uri_col_name] = data[col].apply(create_uri_safe_name)
            except Exception as e:
                print(f"Warning: Could not create URI-safe column for {col}: {e}")

    # Add all new columns at once to avoid fragmentation
    if new_columns:
        # Convert new column dict to dataframe
        new_df = pd.DataFrame(new_columns, index=data.index)

        # Concatenate with original dataframe
        data = pd.concat([data, new_df], axis=1)

    # Create a fresh copy to defragment
    data = data.copy()

    # Save output with careful CSV handling
    try:
        output_path = file_path.replace(".csv", output_suffix)
        data.to_csv(output_path, index=False, na_rep='',
                    quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"Saved cleaned data to {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")

    return data


def process_meme_data(file_paths, index_key_file_path=None, combine=True,
                      combined_output="all_memes_combined_clean.csv"):
    """Process multiple meme data files and optionally combine them"""
    processed_data = {}

    # Process each file
    for file_path in file_paths:
        data = process_meme_file(file_path, index_key_file_path)
        if data is not None:
            file_name = os.path.basename(file_path).replace(".csv", "")
            processed_data[file_name] = data

            # Validate the cleaned file
            output_path = file_path.replace(".csv", "_clean.csv")
            if os.path.exists(output_path):
                validate_cleaned_data(output_path)
                # Check for CSV overflow issues
                check_csv_overflow(output_path)

    # Combine data if requested and if we have processed files
    if combine and len(processed_data) > 0:
        print(f"\nCombining {len(processed_data)} processed files...")

        # Combine all dataframes
        all_dfs = list(processed_data.values())
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Remove any duplicate rows based on ID and Title
        if 'ID' in combined_df.columns and 'Title' in combined_df.columns:
            original_len = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['ID', 'Title'], keep='first')
            deduped_len = len(combined_df)
            if original_len > deduped_len:
                print(f"Removed {original_len - deduped_len} duplicate rows")

        # Save combined file with very careful quoting
        try:
            combined_df.to_csv(combined_output, index=False, na_rep='',
                               quoting=csv.QUOTE_ALL, escapechar='\\')
            print(f"Combined data ({len(combined_df)} rows) saved to {combined_output}")

            # Validate the combined file
            validate_cleaned_data(combined_output)
            # Check for CSV overflow issues
            check_csv_overflow(combined_output)
        except Exception as e:
            print(f"Error saving combined file: {e}")

        return combined_df

    return processed_data