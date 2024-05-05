use crate::{sparse_coo::SparseCOOPrimitive, sparse_csr::SparseCSRPrimitive};

#[derive(Debug, Default, Clone)]
pub struct SparseCSR;

#[derive(Debug, Default, Clone)]
pub struct SparseCOO;

pub trait SparseRepresentation: Clone + Default + Send + Sync + 'static + core::fmt::Debug {
    type Primitive: Clone + Send + 'static + core::fmt::Debug;

    fn name() -> String;
}

impl SparseRepresentation for SparseCOO {
    type Primitive = SparseCOOPrimitive;

    fn name() -> String {
        "SparseCOO".to_owned()
    }
}

impl SparseRepresentation for SparseCSR {
    type Primitive = SparseCSRPrimitive;

    fn name() -> String {
        "SparseCSR".to_owned()
    }
}
