use crate::backend::{Backend, SparseBackend};

use super::{Bool, Dense, Float, Int, Sparse, TensorPrimitive, COO};

pub trait KindRepr<B: Backend>: Clone + core::fmt::Debug {
    /// The primitive type of the tensor.
    type Primitive: Clone + core::fmt::Debug + Send + Sync;

    fn name() -> &'static str;
}

impl<B: Backend> KindRepr<B> for (Float, Dense) {
    type Primitive = TensorPrimitive<B>;

    fn name() -> &'static str {
        "Dense Float"
    }
}

impl<B: Backend> KindRepr<B> for (Int, Dense) {
    type Primitive = B::IntTensorPrimitive;

    fn name() -> &'static str {
        "Dense Int"
    }
}

impl<B: Backend> KindRepr<B> for (Bool, Dense) {
    type Primitive = B::BoolTensorPrimitive;

    fn name() -> &'static str {
        "Dense Bool"
    }
}

impl<B: SparseBackend<COO>> KindRepr<B> for (Float, Sparse<COO>) {
    type Primitive = B::FloatSparseTensorPrimitive;

    fn name() -> &'static str {
        "SparseCOO Float"
    }
}

impl<B: SparseBackend<COO>> KindRepr<B> for (Int, Sparse<COO>) {
    type Primitive = B::IntSparseTensorPrimitive;

    fn name() -> &'static str {
        "SparseCOO Int"
    }
}

impl<B: SparseBackend<COO>> KindRepr<B> for (Bool, Sparse<COO>) {
    type Primitive = B::BoolSparseTensorPrimitive;

    fn name() -> &'static str {
        "SparseCOO Bool"
    }
}
