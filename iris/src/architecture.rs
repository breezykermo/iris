use std::fmt;

// Trait for different architectures
#[derive(Debug, Clone, Copy)]
pub enum HardwareArchitecture {
    SsdStandalone,
    DramRandomPartitioning,
    DramBalancedPartitioning,
}

impl fmt::Display for HardwareArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Delegate formatting to the Debug implementation
        write!(f, "{:?}", self)
    }
}
