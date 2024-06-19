use burn_tensor::{
    backend::Backend, sparse_backend::SparseBackend, Bool, BroadcastArgs, ElementConversion, Float,
    Int, Shape, Tensor,
};

use crate::{SparseCOO, SparseDecorator};

#[derive(Debug, Default, Clone)]
pub struct SparseCOOTensor<B: Backend, const D: usize> {
    coordinates: Vec<B::IntTensorPrimitive<1>>,
    values: B::FloatTensorPrimitive<1>,
}

impl<B> SparseBackend for SparseDecorator<B, SparseCOO>
where
    B: Backend,
{
    type SparseTensorPrimitive<const D: usize> = SparseCOOTensor<B, D>;

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        let device = B::float_device(&dense);
        let shape = B::float_shape(&dense);
        let significant = B::float_not_equal_elem(dense.clone(), 0.elem());

        let coordinates = B::bool_nonzero(significant.clone());

        let number_nonzero = B::int_shape(&coordinates[0]).num_elements();

        let values = B::float_gather(
            0,
            B::float_reshape::<D, 1>(dense, [shape.num_elements()].into()),
            B::bool_nonzero(B::bool_reshape::<D, 1>(
                significant,
                [shape.num_elements()].into(),
            ))
            .remove(0),
        );

        Self::SparseTensorPrimitive {
            coordinates,
            values,
        }
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

    fn sparse_shape<const D: usize>(
        tensor: &burn_tensor::ops::SparseTensor<Self, D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn sparse_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        SparseCOOTensor {
            coordinates: Vec::new(),
            values: B::float_empty(burn_tensor::Shape::new([0]), &device),
        }
    }
}
