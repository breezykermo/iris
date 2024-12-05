use core::panic::PanicMessage;

use oak::fvecs::FlattenedVecs;

/// Inputs: 
/// groundtruth responses for the query 
/// nq * k matrix of ground-truth nearest-neighbors-> num_queries x top-k results for each
/// probably load gt from file through the python script and 
/// ACORN responses for the query
/// 

async fn calculate_recall(gt: Vec<FlattenedVecs>, acorn_index: Vec<FlattenedVecs>, k: usize) {
    // Some way of getting groundtruth -TODO: check FAISS for it
    // Then decide the type of the input. Some Vec of Vec
    nq = gt.len();

    let mut n_1 = 0;
    let mut n_10=0;
    let mut n_100 = 0;

    for i in 0..nq {
        let gt_nn = gt[i*k]; // top 1 search
        for j in 0..k {
            if j <1 {
                n_1 += 1;
            }
            if j < 10 {
                n_10 +=1;
            }
            if j < 100 {
                n_100 += 1;
            }
        }
    }
    println!("R@1 = {:.4}\n", n_1 as f64/ nq as f64);
    println!("R@10 = {:.4}\n", n_10 as f64/ nq as f64);
    println!("R@100 = {:.4}\n", n_100 as f64/ nq as f64);
}