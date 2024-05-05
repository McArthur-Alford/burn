use burn_tensor::{backend::Backend, sparse_backend::SparseBackend};

use crate::{SparseCSR, SparseDecorator, SparseRepresentation};

#[derive(Debug, Default, Clone)]
pub struct SparseCSRPrimitive {}

impl<B> SparseBackend for SparseDecorator<B, SparseCSR>
where
    B: Backend,
{
    type SparseTensorPrimitive<const D: usize> = <SparseCSR as SparseRepresentation>::Primitive;

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_spmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_sddmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_device<const D: usize>(
        tensor: &burn_tensor::ops::SparseTensor<Self, D>,
    ) -> burn_tensor::Device<Self> {
        todo!()
    }

    fn sparse_to_device<const D: usize>(
        tensor: burn_tensor::ops::SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        todo!()
    }
}
