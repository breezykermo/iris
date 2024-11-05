import os
import numpy as np
import struct
import random
import shutil

# Paths to the data files
data_dir = "sift"
base_fname = "sift_base"
query_fname = "sift_query"
base_file = os.path.join(data_dir, f"{base_fname}.fvecs")
query_file = os.path.join(data_dir, f"{query_fname}.fvecs")

# Parameters
attribute_domain = range(1, 13)  # Domain of attribute values (1-12)

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def assign_attributes_and_predicates(base_vectors, query_vectors):
    """Assigns random attributes to base vectors and predicates to query vectors."""
    # Assign a random attribute (1-12) to each base vector
    base_attributes = [random.choice(attribute_domain) for _ in base_vectors]
    
    # Create a query predicate for each query vector, matching a random attribute
    query_predicates = [random.choice(attribute_domain) for _ in query_vectors]
    
    return base_attributes, query_predicates

def main():
    # Read base and query vectors
    print("Reading base vectors...")
    base_vectors = fvecs_read(base_file)
    print("Reading query vectors...")
    query_vectors = fvecs_read(query_file)

    # Assign attributes and predicates
    print("Assigning attributes and predicates...")
    base_attributes, query_predicates = assign_attributes_and_predicates(base_vectors, query_vectors)

    print("Base attributes (sample):", base_attributes[:10])
    print("Query predicates (sample):", query_predicates[:10])

    # Save base attributes
    with open(f"outdir/{base_fname}.csv", "w") as f:
        for attr in base_attributes:
            f.write(f"{attr}\n")

    # Save query predicates
    with open(f"outdir/{query_fname}.csv", "w") as f:
        for pred in query_predicates:
            f.write(f"{pred}\n")

    # Write gitignore 
    with open("outdir/.gitignore", "w") as f:
        f.write("**/*\n!.gitignore")

    # Copy original files
    shutil.copy2(base_file, f"outdir/{base_fname}.fvecs")
    shutil.copy2(query_file, f"outdir/{query_fname}.fvecs")


if __name__ == "__main__":
    main()
