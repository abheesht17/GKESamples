import numpy as np

#  Generate and save synthatic data 
#  1. Run the Python script to create the local .tsv files
# python3 generate_synthetic_data.py

#  2. Upload the generated files to a new directory in your GCS bucket
# gsutil cp train_sample.tsv gs://${BUCKET_NAME}/synthetic_data/train/
# gsutil cp eval_sample.tsv gs://${BUCKET_NAME}/synthetic_data/eval/

NUM_DENSE_FEATURES = 13
VOCAB_SIZES = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36
]
NUM_SPARSE_FEATURES = len(VOCAB_SIZES)

def generate_data(filename, num_samples):
    """Generates synthetic ranking data and saves it to a TSV file."""
    print(f"Generating {num_samples} samples for {filename}...")
    
    with open(filename, "w") as f:
        for _ in range(num_samples):
            label = np.random.randint(0, 2)
            dense_features = np.random.rand(NUM_DENSE_FEATURES)
            sparse_features = [np.random.randint(0, vocab_size) for vocab_size in VOCAB_SIZES]
            all_features = [label] + list(dense_features) + list(sparse_features)
            line = "\t".join(map(str, all_features))
            
            f.write(line + "\n")
            
    print(f"Successfully created {filename}.")

if __name__ == "__main__":
    generate_data("train_sample.tsv", num_samples=100000)
    generate_data("eval_sample.tsv", num_samples=20000)

