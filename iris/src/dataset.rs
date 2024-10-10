// extern crate libc;
use crate::architecture::HardwareArchitecture;
use crate::stubs::gen_random_vecs;
use std::ffi::c_void;

use anyhow::Result;

// Trait for different datasets
pub trait Dataset {
    // Provide basic information about the characteristics of the dataset.
    fn dataset_info(&self) -> String;
    // Returns None if the data is not yet trained, else the HardwareArchitecture on which it was
    // trained.
    fn get_hardware_architecture(&self) -> Option<HardwareArchitecture>;
    // Train the dataset according to a particular architecture. This doesn't actually put the data
    // in the hardware itself, but rather calculates any partitions necessary for that next step.
    // Should set the result of `get_hardware_architecture`, and then returns the appropriate
    // partitions for the target architecture.
    fn train(&mut self, hardware_architecture: HardwareArchitecture) -> Result<Vec<DataPartition>>;

    // Assuming a trained dataset, loads the data onto the actual hardware.
    fn load(&mut self, partitions: Vec<DataPartition>) -> Result<()>;
}

const DIMSTANDARD: usize = 100;
const ONE_THOUSAND: usize = 1_000;

const VDIMS: usize = DIMSTANDARD;
const VSIZE: usize = ONE_THOUSAND;
const VLEN: usize = VDIMS * VSIZE;

#[derive(Clone)]
pub struct DataPartition {
    data: Box<[f64]>,
}

impl DataPartition {
    fn new(data: Box<[f64; VDIMS * VSIZE]>) -> Self {
        Self { data }
    }

    // NOTE: not sure if this is the best way to do it
    fn as_c_ptr(self) -> *mut c_void {
        Box::into_raw(self.data.into_vec().into_boxed_slice()) as *mut c_void
    }
}

/// Deep1B Dataset implementation
pub struct StubVectorDataset {
    pub data: DataPartition,
    hardware_architecture: Option<HardwareArchitecture>,
}

impl StubVectorDataset {
    pub fn new() -> Self {
        // TODO: replace this stub data with the actual data.
        let underlying_data = gen_random_vecs::<VDIMS, VLEN>();

        Self {
            data: DataPartition::new(underlying_data),
            hardware_architecture: None,
        }
    }
}

impl Dataset for StubVectorDataset {
    fn dataset_info(&self) -> String {
        format!("Deep1B stub with {:?} vectors", VSIZE)
    }

    fn train(&mut self, hardware_architecture: HardwareArchitecture) -> Result<Vec<DataPartition>> {
        match hardware_architecture {
            hw @ HardwareArchitecture::SsdStandalone => {
                self.hardware_architecture = Some(hw);
                Ok(vec![self.data.clone()])
            }
            HardwareArchitecture::DramBalancedPartitioning => unimplemented!(),
            HardwareArchitecture::DramRandomPartitioning => unimplemented!(),
        }
    }

    fn get_hardware_architecture(&self) -> Option<HardwareArchitecture> {
        self.hardware_architecture
    }

    fn load(&mut self, partitions: Vec<DataPartition>) -> Result<()> {
        unimplemented!()
    }
}
