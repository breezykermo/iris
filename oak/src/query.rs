// Trait for different loads
pub trait Load {
    fn load_info(&self) -> String;
}

// SyncQueries load implementation
pub struct SyncQueries {
    pub num_queries: usize, // e.g., 10k queries
}

impl Load for SyncQueries {
    fn load_info(&self) -> String {
        format!("{} synchronous queries", self.num_queries)
    }
}
