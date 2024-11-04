use std::fmt;
use std::str::FromStr;

// Trait for different architectures
#[derive(Debug, Clone, Copy)]
pub enum HardwareArchitecture {
    SsdStandalone,
    DramRandomPartitioning,
    DramBalancedLshPartitioned,
    DramBalancedHnswPartitioned,
}

impl FromStr for HardwareArchitecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "SsdReplicated" => Ok(HardwareArchitecture::SsdStandalone),
            "RandomPartitioned" => Ok(HardwareArchitecture::DramRandomPartitioning),
            "BalancedLshPartitioned" => Ok(HardwareArchitecture::DramBalancedLshPartitioned),
            "BalancedHnswPartitioned" => Ok(HardwareArchitecture::DramBalancedHnswPartitioned),
            _ => Err(format!("Invalid hardware architecture: {}", s)),
        }
    }
}

impl fmt::Display for HardwareArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Delegate formatting to the Debug implementation
        write!(f, "{:?}", self)
    }
}
