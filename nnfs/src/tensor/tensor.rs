use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

//
// Helper functions for index arithmetic
//

/// Computes row–major strides for the given shape.
/// For example, shape [a, b, c] yields strides [b*c, c, 1].
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Converts a flat index into a multi–index given a shape (assuming row–major order).
fn multi_index_from_flat(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = compute_strides(shape);
    let mut indices = Vec::with_capacity(shape.len());
    for &stride in &strides {
        indices.push(flat / stride);
        flat %= stride;
    }
    indices
}

/// Computes the dot product between two vectors of the same length.
fn dot_product(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(a, b)| a * b).sum()
}

/// Given two shapes, compute the broadcasted shape according to NumPy–like rules.
/// (Each dimension is compatible if they are equal or one of them is 1.)
fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let rank1 = shape1.len();
    let rank2 = shape2.len();
    let rank = std::cmp::max(rank1, rank2);
    let mut result = Vec::with_capacity(rank);
    for i in 0..rank {
        let dim1 = if i < rank - rank1 {
            1
        } else {
            shape1[i - (rank - rank1)]
        };
        let dim2 = if i < rank - rank2 {
            1
        } else {
            shape2[i - (rank - rank2)]
        };
        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            panic!("Shapes {:?} and {:?} are not broadcastable", shape1, shape2);
        }
        result.push(std::cmp::max(dim1, dim2));
    }
    result
}

//
// The Generic Tensor Type
//

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T> Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    /// Creates a new tensor from a shape and a flat vector of data.
    ///
    /// # Panics
    ///
    /// Panics if the number of elements in `data` does not equal the product of `shape`.
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
        let expected: usize = shape.iter().product();
        assert!(
            data.len() == expected,
            "Data length {} does not match expected size {} for shape {:?}",
            data.len(),
            expected,
            shape
        );
        Tensor { shape, data }
    }

    /// Returns a reference to the tensor's shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns a reference to the tensor's raw data.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Computes the flat index from a full multi–index (assuming the index length equals rank).
    ///
    /// # Panics
    ///
    /// Panics if an index is out of bounds.
    pub fn compute_index(&self, indices: &[usize]) -> usize {
        assert!(
            indices.len() == self.shape.len(),
            "Expected {} indices but got {}",
            self.shape.len(),
            indices.len()
        );
        let strides = compute_strides(&self.shape);
        for (&i, &dim) in indices.iter().zip(self.shape.iter()) {
            assert!(
                i < dim,
                "Index {} out of bounds for dimension size {}",
                i,
                dim
            );
        }
        dot_product(indices, &strides)
    }

    /// Returns a reference to the element at the specified multi–index.
    pub fn get(&self, indices: &[usize]) -> &T {
        let idx = self.compute_index(indices);
        &self.data[idx]
    }

    /// Returns a mutable reference to the element at the specified multi–index.
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let idx = self.compute_index(indices);
        &mut self.data[idx]
    }

    /// For broadcasting: given a multi–index into the broadcasted (larger) shape,
    /// compute the corresponding element from self.
    ///
    /// The rule is that missing (or singleton) dimensions always index element 0.
    fn get_broadcast(&self, broadcast_index: &[usize], broadcast_shape: &[usize]) -> T {
        let r = broadcast_shape.len();
        let k = self.shape.len();
        let strides = compute_strides(&self.shape);
        let mut indices = vec![0; k];
        // Align self.shape to the right of broadcast_shape.
        for i in 0..k {
            let bdim = i + (r - k);
            // If self's dimension is 1, then its only valid index is 0.
            if self.shape[i] == 1 {
                indices[i] = 0;
            } else {
                indices[i] = broadcast_index[bdim];
            }
        }
        let flat_index = dot_product(&indices, &strides);
        self.data[flat_index]
    }

    /// General element–wise operation that supports broadcasting.
    /// The closure `op` is applied to corresponding elements.
    fn broadcast_elemwise<F>(&self, other: &Self, op: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        let bshape = broadcast_shape(&self.shape, &other.shape);
        let total: usize = bshape.iter().product();
        let mut result_data = Vec::with_capacity(total);
        for i in 0..total {
            let b_index = multi_index_from_flat(i, &bshape);
            let a_val = self.get_broadcast(&b_index, &bshape);
            let b_val = other.get_broadcast(&b_index, &bshape);
            result_data.push(op(a_val, b_val));
        }
        Tensor {
            shape: bshape,
            data: result_data,
        }
    }

    /// Matrix multiplication (dot product) for two–dimensional tensors.
    /// If self is (m x n) and other is (n x p), returns (m x p).
    pub fn matmul(&self, other: &Self) -> Self {
        assert!(
            self.shape.len() == 2 && other.shape.len() == 2,
            "matmul requires two 2D tensors (matrices)"
        );
        let m = self.shape[0];
        let n = self.shape[1];
        let p = other.shape[1];
        assert!(
            n == other.shape[0],
            "Inner dimensions must agree for matmul ({} vs {})",
            n,
            other.shape[0]
        );
        let mut result = vec![T::default(); m * p];
        for i in 0..m {
            for j in 0..p {
                let mut sum = T::default();
                for k in 0..n {
                    let a = *self.get(&[i, k]);
                    let b = *other.get(&[k, j]);
                    sum = sum + a * b;
                }
                result[i * p + j] = sum;
            }
        }
        Tensor {
            shape: vec![m, p],
            data: result,
        }
    }

    /// Transposes the tensor.
    ///
    /// If `axes` is provided, it should be a permutation of 0..rank.
    /// Otherwise, the axes are reversed.
    pub fn transpose(&self, axes: Option<&[usize]>) -> Self {
        let rank = self.shape.len();
        let perm: Vec<usize> = match axes {
            Some(axes) => {
                assert!(
                    axes.len() == rank,
                    "Permutation length {} must equal tensor rank {}",
                    axes.len(),
                    rank
                );
                // Ensure it is a permutation.
                let mut seen = vec![false; rank];
                for &axis in axes {
                    assert!(axis < rank, "Axis {} out of bounds", axis);
                    assert!(!seen[axis], "Duplicate axis {} in permutation", axis);
                    seen[axis] = true;
                }
                axes.to_vec()
            }
            None => (0..rank).rev().collect(),
        };

        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        let total = self.data.len();
        let mut new_data = vec![T::default(); total];

        let old_strides = compute_strides(&self.shape);
        let new_strides = compute_strides(&new_shape);

        for i in 0..total {
            let new_multi_index = multi_index_from_flat(i, &new_shape);
            // Invert the permutation: the value in position k of new_multi_index
            // comes from axis perm[k] in the original tensor.
            let mut old_multi_index = vec![0; rank];
            for (k, &axis) in perm.iter().enumerate() {
                old_multi_index[axis] = new_multi_index[k];
            }
            let old_index = dot_product(&old_multi_index, &old_strides);
            new_data[i] = self.data[old_index];
        }

        Tensor {
            shape: new_shape,
            data: new_data,
        }
    }

    /// Reshapes the tensor into a new shape.
    ///
    /// # Panics
    ///
    /// Panics if the total number of elements does not remain the same.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let expected: usize = new_shape.iter().product();
        assert!(
            expected == self.data.len(),
            "New shape {:?} has {} elements but data has {}",
            new_shape,
            expected,
            self.data.len()
        );
        Tensor {
            shape: new_shape,
            data: self.data.clone(),
        }
    }

    /// Reduction: Sum along a specified axis.
    ///
    /// The resulting tensor has the given axis removed.
    pub fn sum(&self, axis: usize) -> Self {
        assert!(axis < self.shape.len(), "Axis {} out of range", axis);
        let mut new_shape = self.shape.clone();
        let n = new_shape.remove(axis);
        let result_len: usize = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let mut multi_index = multi_index_from_flat(i, &new_shape);
            multi_index.insert(axis, 0);
            let mut acc = T::default();
            for j in 0..n {
                multi_index[axis] = j;
                let idx = self.compute_index(&multi_index);
                acc = acc + self.data[idx];
            }
            result_data.push(acc);
        }
        Tensor {
            shape: new_shape,
            data: result_data,
        }
    }

    /// Reduction: Mean along a specified axis.
    ///
    /// **Note:** For simplicity we require that T supports division from f64 via `From<f64>`.
    pub fn mean(&self, axis: usize) -> Self
    where
        T: From<f64>,
    {
        let summed = self.sum(axis);
        let count = self.shape[axis] as f64;
        let data = summed.data.iter().map(|&x| x / T::from(count)).collect();
        Tensor {
            shape: summed.shape,
            data,
        }
    }

    /// Reduction: Maximum value along a specified axis.
    pub fn max(&self, axis: usize) -> Self {
        assert!(axis < self.shape.len(), "Axis {} out of range", axis);
        let mut new_shape = self.shape.clone();
        let n = new_shape.remove(axis);
        let result_len: usize = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let mut multi_index = multi_index_from_flat(i, &new_shape);
            multi_index.insert(axis, 0);
            let mut max_val = *self.get(&multi_index);
            for j in 1..n {
                multi_index[axis] = j;
                let idx = self.compute_index(&multi_index);
                if self.data[idx] > max_val {
                    max_val = self.data[idx];
                }
            }
            result_data.push(max_val);
        }
        Tensor {
            shape: new_shape,
            data: result_data,
        }
    }

    /// Reduction: Minimum value along a specified axis.
    pub fn min(&self, axis: usize) -> Self {
        assert!(axis < self.shape.len(), "Axis {} out of range", axis);
        let mut new_shape = self.shape.clone();
        let n = new_shape.remove(axis);
        let result_len: usize = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(result_len);
        for i in 0..result_len {
            let mut multi_index = multi_index_from_flat(i, &new_shape);
            multi_index.insert(axis, 0);
            let mut min_val = *self.get(&multi_index);
            for j in 1..n {
                multi_index[axis] = j;
                let idx = self.compute_index(&multi_index);
                if self.data[idx] < min_val {
                    min_val = self.data[idx];
                }
            }
            result_data.push(min_val);
        }
        Tensor {
            shape: new_shape,
            data: result_data,
        }
    }

    /// Returns a slice of the tensor along a specified axis at the given index.
    /// For example, for a matrix (2D tensor):
    /// - slice_along_axis(0, i) returns row i
    /// - slice_along_axis(1, j) returns column j
    pub fn slice_along_axis(&self, axis: usize, index: usize) -> Self {
        assert!(axis < self.shape.len(), "Axis {} out of range", axis);
        assert!(
            index < self.shape[axis],
            "Index {} out of bounds for axis {} with size {}",
            index,
            axis,
            self.shape[axis]
        );

        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        let result_len: usize = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(result_len);

        // Calculate strides for the original tensor
        let strides = compute_strides(&self.shape);

        // Create a multi-index for the slice, with the given index at the specified axis
        let mut multi_index = vec![0; self.shape.len()];
        multi_index[axis] = index;

        // Iterate through the new shape and compute the corresponding indices in the original tensor
        for i in 0..result_len {
            let mut current_multi_index = multi_index_from_flat(i, &new_shape);

            // Merge the current multi-index with the slice multi-index
            let mut full_multi_index = Vec::new();
            let mut current_index_iter = current_multi_index.into_iter();
            for j in 0..self.shape.len() {
                if j == axis {
                    full_multi_index.push(multi_index[j]);
                } else {
                    full_multi_index.push(current_index_iter.next().unwrap());
                }
            }

            // Compute the flat index and get the value
            let flat_idx = self.compute_index(&full_multi_index);
            result_data.push(self.data[flat_idx]);
        }

        Tensor {
            shape: new_shape,
            data: result_data,
        }
    }

    /// Returns a row of a matrix (2D tensor) as a vector.
    /// Panics if the tensor is not 2D or if the row index is out of bounds.
    pub fn row(&self, row: usize) -> Self {
        assert!(
            self.shape.len() == 2,
            "row() can only be called on 2D tensors (matrices)"
        );
        self.slice_along_axis(0, row)
    }

    /// Returns a column of a matrix (2D tensor) as a vector.
    /// Panics if the tensor is not 2D or if the column index is out of bounds.
    pub fn column(&self, col: usize) -> Self {
        assert!(
            self.shape.len() == 2,
            "column() can only be called on 2D tensors (matrices)"
        );
        self.slice_along_axis(1, col)
    }
}

//
// Operator Overloads for Element–wise Arithmetic (with broadcasting)
//

impl<T> Add for Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.broadcast_elemwise(&rhs, |a, b| a + b)
    }
}

impl<T> Sub for Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.broadcast_elemwise(&rhs, |a, b| a - b)
    }
}

impl<T> Mul for Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.broadcast_elemwise(&rhs, |a, b| a * b)
    }
}

impl<T> Div for Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self.broadcast_elemwise(&rhs, |a, b| a / b)
    }
}

/// Allow indexing a tensor with a slice of indices.
impl<T> Index<&[usize]> for Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    type Output = T;
    fn index(&self, indices: &[usize]) -> &Self::Output {
        self.get(indices)
    }
}

/// Allow mutable indexing a tensor with a slice of indices.
impl<T> IndexMut<&[usize]> for Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    fn index_mut(&mut self, indices: &[usize]) -> &mut Self::Output {
        self.get_mut(indices)
    }
}

//
// Unit Tests
//
#[cfg(test)]
mod tests {
    use super::*;

    // --- Tensor Tests ---
    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_tensor_invalid_shape() {
        // Data length does not match 2 x 2 = 4.
        let _ = Tensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_indexing() {
        let t = Tensor::from_vec(vec![2, 2], vec![10, 20, 30, 40]);
        assert_eq!(*t.get(&[0, 0]), 10);
        assert_eq!(*t.get(&[1, 1]), 40);
    }

    #[test]
    fn test_tensor_broadcast_addition() {
        let t1 = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        // A "bias" vector that should be broadcasted to shape [2, 2]
        let bias = Tensor::from_vec(vec![2], vec![10.0, 20.0]);
        let result = t1 + bias;
        assert_eq!(result.shape(), &[2, 2]);
        // Expected broadcast addition (row-wise):
        // First row: [1+10, 2+20] = [11, 22]
        // Second row: [3+10, 4+20] = [13, 24]
        assert_eq!(result.data, &[11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_matmul() {
        // Matrix A: 2x3
        let a = Tensor::from_vec(
            vec![2, 3],
            vec![
                1.0, 2.0, 3.0, // Row 0
                4.0, 5.0, 6.0, // Row 1
            ],
        );
        // Matrix B: 3x2
        let b = Tensor::from_vec(
            vec![3, 2],
            vec![
                7.0, 8.0, // Row 0
                9.0, 10.0, // Row 1
                11.0, 12.0, // Row 2
            ],
        );
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        // Expected result:
        // c[0,0] = 1*7 + 2*9 + 3*11 = 58
        // c[0,1] = 1*8 + 2*10 + 3*12 = 64
        // c[1,0] = 4*7 + 5*9 + 6*11 = 139
        // c[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert_eq!(*c.get(&[0, 0]), 58.0);
        assert_eq!(*c.get(&[0, 1]), 64.0);
        assert_eq!(*c.get(&[1, 0]), 139.0);
        assert_eq!(*c.get(&[1, 1]), 154.0);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let t_trans = t.transpose(None);
        assert_eq!(t_trans.shape(), &[3, 2]);
        // Transposed data should be:
        // [1, 4, 2, 5, 3, 6]
        assert_eq!(t_trans.data, &[1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let t_reshaped = t.reshape(vec![3, 2]);
        assert_eq!(t_reshaped.shape(), &[3, 2]);
        // Data order remains the same (row–major order)
        assert_eq!(t_reshaped.data, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reductions() {
        let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let sum_axis0 = t.sum(0);
        // Sum along rows: expected shape [2] and values: [1+3, 2+4] = [4, 6]
        assert_eq!(sum_axis0.shape(), &[2]);
        assert_eq!(sum_axis0.data, &[4.0, 6.0]);

        // Mean along axis 1: expected shape [2] and values: [(1+2)/2, (3+4)/2] = [1.5, 3.5]
        let mean_axis1 = t.mean(1);
        assert_eq!(mean_axis1.shape(), &[2]);
        assert_eq!(mean_axis1.data, &[1.5, 3.5]);

        let max_axis0 = t.max(0);
        assert_eq!(max_axis0.shape(), &[2]);
        assert_eq!(max_axis0.data, &[3.0, 4.0]);

        let min_axis1 = t.min(1);
        assert_eq!(min_axis1.shape(), &[2]);
        assert_eq!(min_axis1.data, &[1.0, 3.0]);
    }

    #[test]
    fn test_matrix_rows_and_columns() {
        let matrix = Tensor::from_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);

        // Test row access
        let row0 = matrix.row(0);
        assert_eq!(row0.shape(), &[3]);
        assert_eq!(row0.data(), &[1, 2, 3]);

        let row1 = matrix.row(1);
        assert_eq!(row1.shape(), &[3]);
        assert_eq!(row1.data(), &[4, 5, 6]);

        // Test column access
        let col0 = matrix.column(0);
        assert_eq!(col0.shape(), &[2]);
        assert_eq!(col0.data(), &[1, 4]);

        let col1 = matrix.column(1);
        assert_eq!(col1.shape(), &[2]);
        assert_eq!(col1.data(), &[2, 5]);

        let col2 = matrix.column(2);
        assert_eq!(col2.shape(), &[2]);
        assert_eq!(col2.data(), &[3, 6]);
    }

    #[test]
    fn test_slice_along_axis() {
        let tensor = Tensor::from_vec(vec![2, 2, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        // Get a slice along axis 0 (should be a 2x3 matrix)
        let slice0 = tensor.slice_along_axis(0, 1);
        assert_eq!(slice0.shape(), &[2, 3]);
        assert_eq!(slice0.data(), &[7, 8, 9, 10, 11, 12]);

        // Get a slice along axis 1 (should be a 2x3 matrix)
        let slice1 = tensor.slice_along_axis(1, 0);
        assert_eq!(slice1.shape(), &[2, 3]);
        assert_eq!(slice1.data(), &[1, 2, 3, 7, 8, 9]);
    }
}
