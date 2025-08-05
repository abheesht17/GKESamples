import tensorflow as tf
import sys
import numpy as np

# Get the file pattern from the command line arguments
if len(sys.argv) < 2:
    print("Usage: python debug_dataset.py <file_pattern>")
    sys.exit(1)

file_pattern = sys.argv[1]
print(f"--- Analyzing files matching: {file_pattern} ---")

try:
    # Create a dataset from the files
    # Use list_files to handle wildcards, then create the TFRecordDataset
    files = tf.data.Dataset.list_files(file_pattern)
    raw_dataset = tf.data.TFRecordDataset(files)

    # Take just one raw record
    for raw_record in raw_dataset.take(1):
        print("\n--- Found one raw record. Parsing... ---")
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        print("\n--- Features found in the record: ---")
        # Iterate over all features in the parsed example
        for key, feature in example.features.feature.items():
            # Determine the type of the feature
            if feature.HasField('bytes_list'):
                kind = 'bytes_list'
                # Decode the first byte string to see its integer representation
                try:
                    first_val = np.frombuffer(feature.bytes_list.value[0], dtype=np.int64)
                    value_preview = (
                        f"{len(feature.bytes_list.value)} items, "
                        f"first item as int64: {first_val}"
                    )
                except (IndexError, ValueError):
                    value_preview = "Could not decode byte string."

            elif feature.HasField('float_list'):
                kind = 'float_list'
                value_preview = feature.float_list.value[:5] # Show first 5 values
            elif feature.HasField('int64_list'):
                kind = 'int64_list'
                value_preview = feature.int64_list.value[:5] # Show first 5 values
            else:
                kind = 'empty'
                value_preview = 'N/A'

            print(f"  - Key: '{key}', Type: {kind}, Value Preview: {value_preview}")
        
        print("\n--- Debugging complete. ---")
        break # We only need to inspect the first record

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("This might happen if the file is not a valid TFRecord or is empty.")

