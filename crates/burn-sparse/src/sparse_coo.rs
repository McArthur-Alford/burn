use burn_tensor::{
    backend::Backend, cast::ToElement, sparse_backend::SparseBackend, ElementConversion, Float,
    Int, Shape, Tensor,
};

use crate::{SparseCOO, SparseDecorator};

#[derive(Clone)]
pub struct SparseCOOTensor<B: Backend, const D: usize> {
    pub coordinates: B::IntTensorPrimitive<2>,
    pub values: B::FloatTensorPrimitive<1>,
    pub shape: Shape<D>,
}

impl<B: Backend + Default, const D: usize> Default for SparseCOOTensor<B, D> {
    fn default() -> Self {
        Self {
            shape: Shape::new([0; D]),
            ..Default::default()
        }
    }
}

impl<B: Backend + std::fmt::Debug, const D: usize> std::fmt::Debug for SparseCOOTensor<B, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let a: Tensor<B, 2, Int> = Tensor::from_primitive(self.coordinates.clone());
        let b: Tensor<B, 1, Float> = Tensor::from_primitive(self.values.clone());

        write!(f, "\ndims: {}\n", a)?;
        write!(f, "values: {}", b)
    }
}

impl<B> SparseBackend for SparseDecorator<B, SparseCOO>
where
    B: Backend,
{
    type SparseTensorPrimitive<const D: usize> = SparseCOOTensor<B, D>;

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        let shape = B::float_shape(&dense);
        let significant = B::float_not_equal_elem(dense.clone(), 0.elem());

        let coordinates = B::bool_nonzero(significant.clone())
            .into_iter()
            .map(|tensor| {
                let length = B::int_shape(&tensor).dims[0];
                let shape = Shape::new([1, length]);
                B::int_reshape(tensor, shape)
            })
            .collect();

        let coordinates = B::int_cat(coordinates, 0);

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
            shape,
        }
    }

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
        } = sparse;

        let device = B::int_device(&coordinates);

        let mut dense = B::float_zeros(shape.clone(), &device);

        let num_nonzero = B::int_shape(&coordinates).dims[1];
        let num_dims = B::int_shape(&coordinates).dims[0];

        for i in 0..num_nonzero {
            let coord = B::int_slice(coordinates.clone(), [0..num_dims, i..i + 1]);
            let coord = B::int_into_data(coord);
            let coord = coord.read().value;

            let value = B::float_slice(values.clone(), [i..i + 1]);
            let value = B::float_reshape(value, Shape::new([1; D]));

            let slice: [_; D] = (0..D)
                .map(|i| coord[i].to_usize()..coord[i].to_usize() + 1)
                .collect::<Vec<_>>()
                .try_into()
                .expect("Invalid Dimensions");

            dense = B::float_slice_assign(dense, slice, value);
        }

        dense
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
        B::float_device(&tensor.values)
    }

    fn sparse_to_device<const D: usize>(
        tensor: burn_tensor::ops::SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_shape<const D: usize>(
        tensor: &Self::SparseTensorPrimitive<D>,
    ) -> burn_tensor::Shape<D> {
        tensor.shape.clone()
    }

    fn sparse_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        SparseCOOTensor {
            coordinates: B::int_empty(burn_tensor::Shape::new([0, 0]), &device),
            values: B::float_empty(burn_tensor::Shape::new([0]), &device),
            shape,
        }
    }

    fn sparse_slice<const D1: usize, const D2: usize>(
        tensor: Self::SparseTensorPrimitive<D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> burn_tensor::ops::SparseTensor<Self, D1> {
        let SparseCOOTensor {
            coordinates,
            values,
            shape,
        } = tensor;

        // tracking values to be kept:
        let device = B::int_device(&coordinates);
        let number_nonzero = B::int_shape(&coordinates).dims[1];
        let mut mask = B::int_ones(Shape::new([number_nonzero]), &device);

        for (dim, bound) in indices.iter().enumerate() {
            let coords = B::int_slice(coordinates.clone(), [dim..dim + 1, 0..number_nonzero]);
            let coords = B::int_reshape(coords, Shape::new([number_nonzero]));

            let mask_lower = B::int_lower_elem(coords.clone(), B::IntElem::from_elem(bound.end));
            let mask_lower = B::bool_into_int(mask_lower);

            let mask_upper = B::int_greater_equal_elem(coords, B::IntElem::from_elem(bound.start));
            let mask_upper = B::bool_into_int(mask_upper);

            mask = B::int_mul(mask, mask_lower);
            mask = B::int_mul(mask, mask_upper);
        }

        let nonzero = B::int_not_equal_elem(mask, B::IntElem::from_elem(0));
        let nonzero = B::bool_nonzero(nonzero);

        let indices_dim1 = nonzero.get(0).expect("Expected the dimension to exist");

        let coordinates = B::int_select(coordinates, 1, indices_dim1.clone());
        let values = B::float_select(values, 0, indices_dim1.clone());

        SparseCOOTensor {
            coordinates,
            values,
            shape,
        }
    }

    fn sparse_into_data<const D: usize>(
        tensor: Self::SparseTensorPrimitive<D>,
    ) -> burn_tensor::Reader<burn_tensor::Data<burn_tensor::ops::FloatElem<Self>, D>> {
        // TODO This is really bad for performance, but necessary to avoid changing the data struct. I guess dont convert massive sparse tensors to data without knowing what your doing?
        let dense = Self::sparse_to_dense(tensor);

        B::float_to_data(&dense)
    }

    fn sparse_from_data<const D: usize>(
        data: burn_tensor::Data<burn_tensor::ops::FloatElem<Self>, D>,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        todo!()
    }
}
