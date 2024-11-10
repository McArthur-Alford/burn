use core::marker::PhantomData;

use crate::backend::{Backend, SparseBackend};

use super::{KindRepr, TensorKind};

/// A type-level representation of the representation of a dense tensor
#[derive(Clone, Debug)]
pub struct Dense;

pub trait SparseRepr: Clone + std::fmt::Debug {
    fn name() -> &'static str;
}

#[derive(Clone, Debug)]
pub struct Sparse<SR: SparseRepr> {
    _marker: PhantomData<SR>,
}

/// A type-level representation of the representation of a sparse CSR tensor
#[derive(Clone, Debug)]
pub struct CSR;

/// A type-level representation of the representation of a sparse COO tensor
#[derive(Clone, Debug)]
pub struct COO;

/// A type-level representation of the representation of a tensor.
pub trait TensorRepr<B: Backend>: Clone + core::fmt::Debug {
    type Primitive<K: TensorKind<B>>: Clone + core::fmt::Debug + Send + Sync
    where
        (K, Self): KindRepr<B>;

    fn name() -> &'static str;
}

impl<B: Backend> TensorRepr<B> for Dense {
    type Primitive<K: TensorKind<B>> = <(K, Self) as KindRepr<B>>::Primitive where (K, Self): KindRepr<B>;

    fn name() -> &'static str {
        "Dense"
    }
}

impl SparseRepr for CSR {
    fn name() -> &'static str {
        "SparseCSR"
    }
}

impl SparseRepr for COO {
    fn name() -> &'static str {
        "SparseCOO"
    }
}

impl<B: Backend + SparseBackend<R>, R: SparseRepr> TensorRepr<B> for Sparse<R> {
    type Primitive<K: TensorKind<B>> = <(K, Self) as KindRepr<B>>::Primitive where (K, Self): KindRepr<B>;

    fn name() -> &'static str {
        R::name()
    }
}
