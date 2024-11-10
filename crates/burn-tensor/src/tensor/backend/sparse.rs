use crate::{SparseRepr, TensorRepr};

use super::Backend;

pub trait SparseBackend<R: TensorRepr<Self> + SparseRepr>: Backend {
    type FloatSparseTensorPrimitive: Clone + Send + Sync + 'static + core::fmt::Debug;
    type IntSparseTensorPrimitive: Clone + Send + Sync + 'static + core::fmt::Debug;
    type BoolSparseTensorPrimitive: Clone + Send + Sync + 'static + core::fmt::Debug;
}
