use burn_tensor::{
    backend::Backend, cast::ToElement, sparse_backend::SparseBackend, Data, ElementConversion,
    Float, Int, Shape, Tensor,
};

use crate::{SparseCOO, SparseDecorator};

#[derive(Clone, Debug)]
pub struct SparseCOOTensor<B: Backend, const D: usize> {
    pub coordinates: B::IntTensorPrimitive<2>,
    pub values: B::FloatTensorPrimitive<1>,
    pub shape: Shape<D>,
}

impl<B: Backend, const D: usize> std::fmt::Display for SparseCOOTensor<B, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coords: Tensor<B, 2, Int> = Tensor::from_primitive(self.coordinates.clone());
        let values: Tensor<B, 1, Float> = Tensor::from_primitive(self.values.clone());

        write!(f, "coords: {}\n", coords)?;
        write!(f, "values: {}", values)
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

        let num_nonzero = B::int_shape(&coordinates).dims[1];
        let device = B::int_device(&coordinates);

        let dense = B::float_zeros(Shape::new([shape.num_elements()]), &device);

        let mut strides_data = [[1]; D];
        for i in (0..D - 1).rev() {
            strides_data[i] = [strides_data[i + 1][0] * shape.dims[i + 1] as i64];
        }

        let strides_data: Data<B::IntElem, 2> = Data::from(strides_data).convert();

        let strides = B::int_from_data(strides_data, &device);

        let coordinates = B::int_mul(strides, coordinates);

        let coordinates = B::int_sum_dim(coordinates, 0);

        let coordinates = B::int_reshape(coordinates, Shape::new([num_nonzero]));

        let dense = B::float_select_assign(dense, 0, coordinates, values);

        B::float_reshape(dense, shape)
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
        SparseCOOTensor {
            coordinates: B::int_to_device(tensor.coordinates, &device),
            values: B::float_to_device(tensor.values, &device),
            shape: tensor.shape,
        }
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
        let dense = Self::sparse_to_dense(tensor);

        B::float_to_data(&dense)
    }

    fn sparse_from_data<const D: usize>(
        data: burn_tensor::Data<burn_tensor::ops::FloatElem<Self>, D>,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        let dense = B::float_from_data(data, &device);
        Self::sparse_to_sparse(dense)
    }
}
