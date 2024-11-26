/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include "oak/third_party/ACORN/faiss/IndexFlat.h"
#include "oak/third_party/ACORN/faiss/IndexPQ.h"
#include "oak/third_party/ACORN/faiss/IndexScalarQuantizer.h"
#include "oak/third_party/ACORN/faiss/impl/ACORN.h"
#include "oak/third_party/ACORN/faiss/utils/utils.h"
// #include "oak/src/lib.rs.h"
#include "rust/cxx.h"

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>

namespace faiss {

struct IndexACORN;




/** The ACORN index is a normal random-access index with a ACORN
 * link structure built on top */

struct IndexACORN : Index {
    typedef ACORN::storage_idx_t storage_idx_t;

    // the link strcuture
    ACORN acorn; // TODO change to hybrid

    // the sequential storage
    bool own_fields;
    Index* storage;

//     ReconstructFromNeighbors* reconstruct_from_neighbors;

    explicit IndexACORN(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2); // defaults d = 0, M=32, gamma=1
    explicit IndexACORN(Index* storage, int M, int gamma, std::vector<int>& metadata, int M_beta);
//     explicit IndexACORN(); // TODO check this is right

    ~IndexACORN() override;

    // add n vectors of dimension d to the index, x is the matrix of vectors
    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    // search for metadata
    // this doesn't override normal search since definition has a filter param - search is overloaded
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            char* filter_id_map,
            const SearchParameters* params = nullptr) const;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    

    // added for debugging
    void printStats(bool print_edge_list=false, bool print_filtered_edge_lists=false, int filter=-1, Operation op=EQUAL);

    private:
        const int debugFlag = 0;

        void debugTime() {
                if (debugFlag) {
                struct timeval tval;
                gettimeofday(&tval, NULL);
                struct tm *tm_info = localtime(&tval.tv_sec);
                char timeBuff[25] = "";
                strftime(timeBuff, 25, "%H:%M:%S", tm_info);
                char timeBuffWithMilli[50] = "";
                sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
                std::string timestamp(timeBuffWithMilli);
                        std::cout << timestamp << std::flush;
        }
        }

        //needs atleast 2 args always
        //  alt debugFlag = 1 // fprintf(stderr, fmt, __VA_ARGS__); 
        #define debug(fmt, ...) \
        do { \
                if (debugFlag == 1) { \
                fprintf(stdout, "--" fmt, __VA_ARGS__);\
                } \
                if (debugFlag == 2) { \
                debugTime(); \
                fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
                } \
        } while (0)



        double elapsed() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + tv.tv_usec * 1e-6;
        }
};

/** Flat index topped with with a ACORN structure to access elements
 *  more efficiently.
 */

struct IndexACORNFlat : IndexACORN {
    IndexACORNFlat();
    IndexACORNFlat(int d, int M, int gamma, std::vector<int>& metadata, int M_beta, MetricType metric = METRIC_L2);

};

// OAK: standalone function to construct a new index from Rust over FFI.
std::unique_ptr<IndexACORNFlat> new_index_acorn(
  int d,
  int M,
  int gamma,
  int M_beta,
  const rust::Vec<int>& metadata
); 

// OAK: standalone function to add vectors to an index from Rust over FFI.
void add_to_index(
  std::unique_ptr<IndexACORNFlat>& idx,
  idx_t n,        // number of vectors to add
  const float* x  // pointer to a contiguous array of the vectors to add
);

// OAK: standalone function to search vectors from an index from Rust over FFI.
void search_index(
  std::unique_ptr<IndexACORNFlat>& idx,
  idx_t n,            // number of query vectors
  const float* x,     // pointer to an array of the query vectors 
  idx_t k,            // number of vectors to return for each query vector
  float* distances,   // pointer to an array of (k*n) floats, each representing a distance of the result from the query vector 
  idx_t* labels,      // pointer to an array of (k*n) indices, each representing the ID of the query vector in idx 
  char* filter_id_map// a bitmap of the IDs in the filter, an array of (n * N) bools, where N is the total number of vectors in the index, and a '1' represents that the vector at that index passes the predicate for that query.
);


} // namespace faiss
