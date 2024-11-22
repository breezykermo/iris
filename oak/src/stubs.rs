use rand::Rng;
// use tracing::info;

pub fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng(); // Initialize the thread-local random number generator
    (0..dim).map(|_| rng.gen()).collect() // Generate 'dim' random numbers
}

// pub fn gen_random_vecs<const VDIM: usize, const VLEN: usize>() -> Box<[f64; VLEN]> {
//     let mut rng = rand::thread_rng();
//     let mut vectors = Box::new([0.0; VLEN]);
//     let quantity = VLEN / VDIM;
//
//     info!(
//         "Generating {} random vectors with {} dimensions...",
//         quantity, VDIM
//     );
//
//     for i in 0..quantity {
//         for j in 0..VDIM {
//             vectors[(i * VDIM) + j] = rng.gen_range(0.0..1.0); // Fill with random numbers
//         }
//     }
//     info!("Random vecs generated.");
//
//     vectors
// }
