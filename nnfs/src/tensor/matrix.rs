use crate::tensor::tensor::Tensor;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

//
// Matrix: A 2D tensor wrapper
//

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    tensor: Tensor<T>,
}

impl<T> Matrix<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    /// Creates a new matrix (2D tensor) from the number of rows, columns, and a flat vector.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Matrix {
            tensor: Tensor::from_vec(vec![rows, cols], data),
        }
    }

    /// Returns the inner tensor struct for this Matrix
    pub fn inner_tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Returns the matrix shape as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.tensor.shape()[0], self.tensor.shape()[1])
    }

    /// Returns the matrix raw data.
    pub fn data(&self) -> &Vec<T> {
        self.tensor.data()
    }

    /// Returns a reference to the element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> &T {
        self.tensor.get(&[row, col])
    }

    /// Returns a mutable reference to the element at (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        self.tensor.get_mut(&[row, col])
    }

    /// Standard matrix multiplication.
    pub fn matmul(&self, other: &Self) -> Self {
        Matrix {
            tensor: self.tensor.matmul(&other.tensor),
        }
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Self {
        Matrix {
            tensor: self.tensor.transpose(None),
        }
    }

    /// Reshapes the matrix.
    /// (Note: reshaping may yield a tensor that is not 2D.)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor<T> {
        self.tensor.reshape(new_shape)
    }

    /// Element–wise addition.
    pub fn add(&self, other: &Self) -> Self {
        Matrix {
            tensor: self.tensor.clone() + other.tensor.clone(),
        }
    }
    // Similarly, you can add methods for sub(), mul(), div(), etc.
}

/// Allow indexing a matrix with (row, col).
impl<T> Index<(usize, usize)> for Matrix<T>
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
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        self.get(row, col)
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        self.get_mut(row, col)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // --- Matrix Tests ---
    #[test]
    fn test_matrix_creation_and_indexing() {
        let m = Matrix::from_vec(2, 2, vec![10, 20, 30, 40]);
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(*m.get(0, 0), 10);
        assert_eq!(*m.get(1, 1), 40);
    }

    #[test]
    fn test_matrix_matmul() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), (2, 2));
        assert_eq!(*c.get(0, 0), 58.0);
        assert_eq!(*c.get(0, 1), 64.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let m_t = m.transpose();
        assert_eq!(m_t.shape(), (3, 2));
        assert_eq!(*m_t.get(0, 0), 1);
        assert_eq!(*m_t.get(2, 1), 6);
    }
}
