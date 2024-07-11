use std::{future::Future, ops::Range};

use crate::backend::SparseBackend;
use burn_tensor::{backend::Backend, BasicOps, Shape, TensorData, TensorKind};

/// A type-level representation of the kind of a sparse (float) tensor.
#[derive(Clone, Debug)]
pub struct Sparse;

impl<B: SparseBackend> TensorKind<B> for Sparse {
    type Primitive<const D: usize> = B::SparseTensorPrimitive<D>;
    fn name() -> &'static str {
        "Sparse"
    }
}

impl<B: SparseBackend> BasicOps<B> for Sparse {
    type Elem = B::FloatElem;

    fn into_data_async<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> impl Future<Output = TensorData> + Send {
        B::sparse_into_data(tensor)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::sparse_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_to_device(tensor, device)
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_from_data(data, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::sparse_shape(tensor)
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_empty(shape, device)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::sparse_slice(tensor, ranges)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        todo!()
    }

    fn transpose<const D: usize>(tensor: Self::Primitive<D>) -> Self::Primitive<D> {
        todo!()
    }

    fn swap_dims<const D: usize>(
        tensor: Self::Primitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn permute<const D: usize>(tensor: Self::Primitive<D>, axes: [usize; D]) -> Self::Primitive<D> {
        todo!()
    }

    fn flip<const D: usize>(tensor: Self::Primitive<D>, axes: &[usize]) -> Self::Primitive<D> {
        todo!()
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        todo!()
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        todo!()
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        todo!()
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn not_equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn any<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, 1, burn_tensor::Bool> {
        todo!()
    }

    fn any_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn all<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> burn_tensor::Tensor<B, 1, burn_tensor::Bool> {
        todo!()
    }

    fn all_dim<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
    ) -> burn_tensor::Tensor<B, D, burn_tensor::Bool> {
        todo!()
    }

    fn expand<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        todo!()
    }
}
