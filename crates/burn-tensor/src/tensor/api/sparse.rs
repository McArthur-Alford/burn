use crate::{backend::Backend, sparse_backend::SparseBackend, Float, Int, Sparse, Tensor};

impl<const D: usize, B> Tensor<B, D, Sparse>
where
    B: SparseBackend,
{
    pub fn dense(self) -> Tensor<B, D, Float> {
        Tensor::new(B::sparse_to_dense(self.primitive))
    }

    pub fn dense_int(self) -> Tensor<B, D, Int> {
        self.dense().int()
    }
}
