use crate::backend::{Backend, SparseBackend};

use super::TensorKind;

/// A type-level representation of the representation of a dense tensor
#[derive(Clone, Debug)]
pub struct Dense;

pub trait SparseRepr: Clone + std::fmt::Debug {
    fn name() -> &'static str;
}

/// A type-level representation of the representation of a sparse CSR tensor
#[derive(Clone, Debug)]
pub struct SparseCSR;

/// A type-level representation of the representation of a sparse COO tensor
#[derive(Clone, Debug)]
pub struct SparseCOO;

/// A type-level representation of the representation of a tensor.
pub trait TensorRepr<B: Backend>: Clone + core::fmt::Debug {
    fn name() -> &'static str;
}

impl<B: Backend> TensorRepr<B> for Dense {
    fn name() -> &'static str {
        "Dense"
    }
}

impl SparseRepr for SparseCSR {
    fn name() -> &'static str {
        "SparseCSR"
    }
}

impl SparseRepr for SparseCOO {
    fn name() -> &'static str {
        "SparseCOO"
    }
}

impl<B: Backend + SparseBackend<R>, R: SparseRepr> TensorRepr<B> for R {
    fn name() -> &'static str {
        R::name()
    }
}
