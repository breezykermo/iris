import os
import numpy as np
import struct
import random
import shutil

# Paths to the data files
data_dir = "siftsmall"
base_fname = "siftsmall_base"
query_fname = "siftsmall_query"
gt_name = "siftsmall_groundtruth"
base_file = os.path.join(data_dir, f"{base_fname}.fvecs")
query_file = os.path.join(data_dir, f"{query_fname}.fvecs")
groundtruth_file = os.path.join(data_dir, f"{gt_name}.ivecs")

single_predicate = 1

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

def generate_groundtruth(gt, base_predicates):
    predicate_gt = []
    no_such_index = [] #query vectors where a nearest neighbor wasn't found
    for i in range(len(gt)):
        # for each query vector, checking for the 
        # nearest neighbor matching our predicate
        nearest_neighbors = gt[i]
        satisfying_nearest_neighbor = -1
        # print("Query number", i)
        for j in range(100):
            pred_index = nearest_neighbors[j]
            # print("NN number", j, base_predicates[pred_index])
            if base_predicates[pred_index] == single_predicate:
                
                satisfying_nearest_neighbor = pred_index
                break
            
        if satisfying_nearest_neighbor == -1:
            no_such_index.append(i)
        else:
            predicate_gt.append(satisfying_nearest_neighbor)
        # if no such found, raise error
    return predicate_gt, no_such_index

def main():
    # Read base and query vectors
    print("Reading base vectors...")
    print(base_file)
    base_vectors = fvecs_read(base_file)
    print("Reading query vectors...")
    print(query_file)
    query_vectors = fvecs_read(query_file)

    # Assign attributes and predicates
    print("Assigning attributes and predicates...")
    base_attributes, query_predicates = assign_attributes_and_predicates(base_vectors, query_vectors)

    print("Base attributes (sample):", base_attributes[:10])
    print("Query predicates (sample):", query_predicates[:10])

    # Save base attributes
    print("Saving to outdir/" + base_fname)
    with open(f"outdir/{base_fname}.csv", "w") as f:
        for attr in base_attributes:
            f.write(f"{attr}\n")

    # Save query predicates
    print("Saving to outdir/" + query_fname)
    with open(f"outdir/{query_fname}.csv", "w") as f:
        for pred in query_predicates:
            f.write(f"{pred}\n")

    # Generate new groundtruth
    print("generating groundtruth...")
    groundtruth = ivecs_read(groundtruth_file)
    predicate_gt, no_NN = generate_groundtruth(groundtruth, base_attributes)
    print("No index found for", len(no_NN), "queries")
    print(no_NN[:10])

    print("saving gt to outdir/"+ gt_name)
    with open(f"outdir/{gt_name}.csv", "w") as f:
        for pred in predicate_gt:
            f.write(f"{pred}\n")


    # Write gitignore 
    # with open("outdir/.gitignore", "w") as f:
    #     f.write("**/*\n!.gitignore")

    # Copy original files
    shutil.copy2(base_file, f"outdir/{base_fname}.fvecs")
    shutil.copy2(query_file, f"outdir/{query_fname}.fvecs")


if __name__ == "__main__":
    main()


# // do fvecsdataset new from query file
# // can call flattened vecs from len 100 and chunks rustvec chunks on underlying array of dimensionality