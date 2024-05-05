use crate::{
    backend::Backend,
    ops::{FloatTensor, SparseTensor},
    Device, Shape,
};

pub trait SparseBackend: Backend {
    type SparseTensorPrimitive<const D: usize>: Clone + Send + 'static + core::fmt::Debug;

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D>;

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D>;

    fn sparse_spmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D>;

    fn sparse_sddmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn sparse_device<const D: usize>(tensor: &SparseTensor<Self, D>) -> Device<Self>;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    fn sparse_to_device<const D: usize>(
        tensor: SparseTensor<Self, D>,
        device: &Device<Self>,
    ) -> SparseTensor<Self, D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn sparse_shape<const D: usize>(tensor: &SparseTensor<Self, D>) -> Shape<D>;
}
