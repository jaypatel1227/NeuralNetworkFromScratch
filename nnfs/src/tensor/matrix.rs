use crate::tensor::tensor::Tensor;
use crate::tensor::vector::Vector;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

//
// Matrix: A 2D tensor wrapper
//

#[derive(Debug, Clone, PartialEq)]
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
    /// Creates a new matrix from a 2D tensor
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2-dimensional
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        assert!(
            tensor.shape().len() == 2,
            "Matrix must be 2-dimensional, got shape {:?}",
            tensor.shape()
        );
        Matrix { tensor }
    }

    /// Creates a new matrix (2D tensor) from the number of rows, columns, and a flat vector.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Matrix {
            tensor: Tensor::from_vec(vec![rows, cols], data),
        }
    }

    /// Creates a new matrix from row-major data with the given number of rows and columns
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        Matrix {
            tensor: Tensor::from_vec(vec![rows, cols], data),
        }
    }

    /// Returns the number of rows in the matrix
    pub fn rows(&self) -> usize {
        self.tensor.shape()[0]
    }

    /// Returns the number of columns in the matrix
    pub fn cols(&self) -> usize {
        self.tensor.shape()[1]
    }

    /// Returns the shape of the matrix as (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Returns a row of the matrix as a vector (1D tensor)
    pub fn row(&self, row: usize) -> Vector<T> {
        Vector::from_tensor(self.tensor.row(row))
    }

    /// Returns a column of the matrix as a vector (1D tensor)
    pub fn column(&self, col: usize) -> Vector<T> {
        Vector::from_tensor(self.tensor.column(col))
    }

    /// Returns the underlying tensor
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Converts this matrix into its underlying tensor
    pub fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }

    pub fn to_vector(&self) -> Vector<T> {
        assert!(
            self.shape().0 == 1 || self.shape().1 == 1,
            "A matrix must be 1 x n or m x 1 to be converted to a Vector."
        );
        Vector::from_vec(self.data().clone())
    }

    /// Returns the matrix raw data.
    pub fn data(&self) -> &Vec<T> {
        &self.tensor.data()
    }

    /// Returns a reference to the element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> &T {
        self.tensor.get(&[row, col])
    }

    /// Returns a mutable reference to the element at (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        self.tensor.get_mut(&[row, col])
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

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Matrix {
            tensor: self.tensor.clone() - other.tensor.clone(),
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        Matrix {
            tensor: self.tensor.matmul(&other.tensor),
        }
    }

    /// Element-wise division.
    pub fn div(&self, other: &Self) -> Self {
        Matrix {
            tensor: self.tensor.clone() / other.tensor.clone(),
        }
    }
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

impl<T> From<Tensor<T>> for Matrix<T> {
    fn from(tensor: Tensor<T>) -> Self {
        Matrix { tensor }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // --- Matrix Tests ---
    #[test]
    fn test_matrix_creation_and_access() {
        let matrix = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(matrix.shape(), (2, 3));

        let row0 = matrix.row(0);
        assert_eq!(row0.len(), 3);
        assert_eq!(row0.as_tensor().data(), &[1, 2, 3]);

        let row1 = matrix.row(1);
        assert_eq!(row1.len(), 3);
        assert_eq!(row1.as_tensor().data(), &[4, 5, 6]);

        let col0 = matrix.column(0);
        assert_eq!(col0.len(), 2);
        assert_eq!(col0.as_tensor().data(), &[1, 4]);

        let col1 = matrix.column(1);
        assert_eq!(col1.len(), 2);
        assert_eq!(col1.as_tensor().data(), &[2, 5]);
    }

    #[test]
    #[should_panic(expected = "Matrix must be 2-dimensional")]
    fn test_invalid_matrix_creation() {
        let tensor = Tensor::from_vec(vec![2], vec![1, 2]); // 1D tensor
        let _matrix = Matrix::from_tensor(tensor);
    }

    #[test]
    fn test_matrix_matmul() {
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.mul(&b);
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
