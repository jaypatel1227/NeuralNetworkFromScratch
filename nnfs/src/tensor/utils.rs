use num_traits::{Float, Num};
use rand::Rng;
use std::ops::{Add, Div, Mul, Sub};

use crate::tensor::matrix::Matrix;
use crate::tensor::tensor::Tensor;
use crate::tensor::vector::Vector;

/// Creates a new tensor filled with zeros.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
pub fn zeros<T>(shape: Vec<usize>) -> Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Num
        + From<u8>,
{
    let size: usize = shape.iter().product();
    let data = vec![T::default(); size];
    Tensor::from_vec(shape, data)
}

/// Creates a new tensor filled with ones.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
pub fn ones<T>(shape: Vec<usize>) -> Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Num
        + From<u8>,
{
    let size: usize = shape.iter().product();
    let data = vec![T::from(1); size];
    Tensor::from_vec(shape, data)
}

/// Creates a new tensor filled with random numbers between -1 and 1.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
pub fn random<T>(shape: Vec<usize>) -> Tensor<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Float
        + From<f64>,
{
    let size: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data: Vec<T> = (0..size)
        .map(|_| {
            let val: f64 = rng.random_range(-1.0..1.0);
            val.into()
        })
        .collect();
    Tensor::from_vec(shape, data)
}

/// Creates a new matrix (2D tensor) filled with zeros.
///
/// # Arguments
///
/// * `rows` - The number of rows in the matrix.
/// * `cols` - The number of columns in the matrix.
pub fn matrix_zeros<T>(rows: usize, cols: usize) -> Matrix<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Num
        + From<u8>,
{
    let shape = vec![rows, cols];
    let size: usize = shape.iter().product();
    let data = vec![T::default(); size];
    Tensor::from_vec(shape, data).into()
}

/// Creates a new matrix (2D tensor) filled with ones.
///
/// # Arguments
///
/// * `rows` - The number of rows in the matrix.
/// * `cols` - The number of columns in the matrix.
pub fn matrix_ones<T>(rows: usize, cols: usize) -> Matrix<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Num
        + From<u8>,
{
    let shape = vec![rows, cols];
    let size: usize = shape.iter().product();
    let data = vec![T::from(1); size];
    Tensor::from_vec(shape, data).into()
}

/// Creates a new matrix (2D tensor) filled with random numbers between -1 and 1.
///
/// # Arguments
///
/// * `rows` - The number of rows in the matrix.
/// * `cols` - The number of columns in the matrix.
pub fn matrix_random<T>(rows: usize, cols: usize) -> Matrix<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Float
        + From<f64>,
{
    let shape = vec![rows, cols];
    let size: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data: Vec<T> = (0..size)
        .map(|_| {
            let val: f64 = rng.random_range(-1.0..1.0);
            val.into()
        })
        .collect();
    Tensor::from_vec(shape, data).into()
}

/// Creates a new vector (1D tensor) filled with zeros.
///
/// # Arguments
///
/// * `size` - The size of the vector.
pub fn vector_zeros<T>(size: usize) -> Vector<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Num
        + From<u8>,
{
    let shape = vec![size];
    let data = vec![T::default(); size];
    Tensor::from_vec(shape, data).into()
}

/// Creates a new vector (1D tensor) filled with ones.
///
/// # Arguments
///
/// * `size` - The size of the vector.
pub fn vector_ones<T>(size: usize) -> Vector<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Num
        + From<u8>,
{
    let shape = vec![size];
    let data = vec![T::from(1); size];
    Tensor::from_vec(shape, data).into()
}

/// Creates a new vector (1D tensor) filled with random numbers between -1 and 1.
///
/// # Arguments
///
/// * `size` - The size of the vector.
pub fn vector_random<T>(size: usize) -> Vector<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Float
        + From<f64>,
{
    let shape = vec![size];
    let size: usize = shape.iter().product();
    let mut rng = rand::rng();
    let data: Vec<T> = (0..size)
        .map(|_| {
            let val: f64 = rng.random_range(-1.0..1.0);
            val.into()
        })
        .collect();
    Tensor::from_vec(shape, data).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t: Tensor<f64> = zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ones() {
        let t: Tensor<i32> = ones(vec![2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.data(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_random() {
        let t: Tensor<f64> = random(vec![1, 5]);
        assert_eq!(t.shape(), &[1, 5]);
        for &val in t.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_matrix_zeros() {
        let m: Matrix<f64> = matrix_zeros(3, 2);
        assert_eq!(m.shape(), (3, 2));
        assert_eq!(m.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_matrix_ones() {
        let m: Matrix<i32> = matrix_ones(2, 3);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.data(), &[1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_matrix_random() {
        let m: Matrix<f64> = matrix_random(2, 2);
        assert_eq!(m.shape(), (2, 2));
        for &val in m.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_vector_zeros() {
        let v: Vector<f64> = vector_zeros(4);
        assert_eq!(v.len(), 4);
        assert_eq!(v.data(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vector_ones() {
        let v: Vector<i32> = vector_ones(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v.data(), &[1, 1, 1]);
    }

    #[test]
    fn test_vector_random() {
        let v: Vector<f64> = vector_random(5);
        assert_eq!(v.len(), 5);
        for &val in v.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
}
