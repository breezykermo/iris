// extern crate libc;
use crate::architecture::HardwareArchitecture;
use crate::stubs::gen_random_vecs;
use std::ffi::c_void;

use anyhow::Result;

// Trait for different datasets
pub trait Dataset {
    // Provide basic information about the characteristics of the dataset.
    fn dataset_info(&self) -> String;
    // Train the dataset according to a particular architecture.
    fn train(&mut self, hardware_architecture: HardwareArchitecture) -> Result<()>;
    // Returns None if the data is not yet trained, else the HardwareArchitecture on which it was
    // trained.
    fn get_hardware_architecture(&self) -> Option<HardwareArchitecture>;
    // Returns None if the data is not yet trained, else the dataset appropriately partitioned.
    fn get_partitions(&self) -> Option<Vec<DataPartition>>;
}

const DIMSTANDARD: usize = 100;
const ONE_THOUSAND: usize = 1_000;

const VDIMS: usize = DIMSTANDARD;
const VSIZE: usize = ONE_THOUSAND;
const VLEN: usize = VDIMS * VSIZE;

#[derive(Clone)]
struct DataPartition {
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
pub struct Deep1B {
    pub data: DataPartition,
    is_single_partition: bool,
    hardware_architecture: Option<HardwareArchitecture>,
}

impl Deep1B {
    pub fn new() -> Self {
        // TODO: replace this stub data with the actual data.
        let underlying_data = gen_random_vecs::<VDIMS, VLEN>();

        Self {
            data: DataPartition::new(underlying_data),
            is_single_partition: false,
            hardware_architecture: None,
        }
    }
}

impl Dataset for Deep1B {
    fn dataset_info(&self) -> String {
        format!("Deep1B with {:?} vectors", VSIZE)
    }

    fn train(&mut self, hardware_architecture: HardwareArchitecture) -> Result<()> {
        match hardware_architecture {
            hw @ HardwareArchitecture::SsdStandalone => {
                self.is_single_partition = true;
                self.hardware_architecture = Some(hw);
                Ok(())
            }
            HardwareArchitecture::DramBalancedPartitioning => unimplemented!(),
            HardwareArchitecture::DramRandomPartitioning => unimplemented!(),
        }
    }

    fn get_hardware_architecture(&self) -> Option<HardwareArchitecture> {
        self.hardware_architecture
    }

    // Once the dataset is trained, this returns the number of partitions.  In the case that there
    // is only one partition, this will have length one.
    fn get_partitions(&self) -> Option<Vec<DataPartition>> {
        if self.is_single_partition {
            Some(vec![self.data.clone()])
        } else {
            None
        }
    }
}
