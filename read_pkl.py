import pickle as pkl
import os
import argparse


def print_sample(sample, index):
    print(f"\n{'='*50}")
    print(f"Sample {index}")
    print(f"{'='*50}")
    print(f"time   : {sample['time']}")
    print(f"action : {sample['action']}")
    print(f"\n--- State ---")
    for key, val in sample['state'].items():
        if val is not None:
            print(f"  {key}: {val}")


def read_pkl(file_path, num_samples=1, sample_index=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'rb') as f:
        samples = pkl.load(f)

    print(f"File: {file_path}")
    print(f"Total samples: {len(samples)}")

    start = sample_index if sample_index is not None else 0
    if start >= len(samples):
        print(f"Index {start} out of range (max {len(samples)-1})")
        return

    for i in range(start, min(start + num_samples, len(samples))):
        print_sample(samples[i], i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and inspect a CoLight .pkl file")
    parser.add_argument("file", type=str, help="Path to the .pkl file")
    parser.add_argument("-n", type=int, default=1, help="Number of samples to print (default: 1)")
    parser.add_argument("-i", type=int, default=None, help="Print a specific sample by index")
    args = parser.parse_args()

    read_pkl(args.file, num_samples=args.n, sample_index=args.i)
