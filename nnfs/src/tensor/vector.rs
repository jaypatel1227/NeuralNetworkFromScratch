use crate::tensor::tensor::Tensor;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// A Vector is a 1-dimensional Tensor
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T> {
    tensor: Tensor<T>,
}

impl<T> Vector<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    /// Creates a new vector from a 1D tensor
    /// 
    /// # Panics
    /// 
    /// Panics if the tensor is not 1-dimensional
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        assert!(
            tensor.shape().len() == 1,
            "Vector must be 1-dimensional, got shape {:?}",
            tensor.shape()
        );
        Vector { tensor }
    }

    /// Creates a new vector from data
    pub fn from_vec(data: Vec<T>) -> Self {
        Vector {
            tensor: Tensor::from_vec(vec![data.len()], data),
        }
    }

    /// Returns the length of the vector
    pub fn len(&self) -> usize {
        self.tensor.shape()[0]
    }

    /// Returns true if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the underlying tensor
    pub fn as_tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Converts this vector into its underlying tensor
    pub fn into_tensor(self) -> Tensor<T> {
        self.tensor
    }

    /// Returns a reference to the element at the given index
    pub fn get(&self, index: usize) -> &T {
        self.tensor.get(&[index])
    }

    /// Returns a mutable reference to the element at the given index
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        self.tensor.get_mut(&[index])
    }

    /// Returns a reference to the vector's raw data
    pub fn data(&self) -> &Vec<T> {
        self.tensor.data()
    }

    /// Computes the dot product with another vector
    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector dimensions must match for dot product"
        );
        let mut result = T::default();
        for i in 0..self.len() {
            result = result + (*self.get(i) * *other.get(i));
        }
        result
    }

    /// Computes the element-wise sum with another vector
    pub fn add(&self, other: &Self) -> Self {
        Vector {
            tensor: self.tensor.clone() + other.tensor.clone(),
        }
    }

    /// Computes the element-wise difference with another vector
    pub fn sub(&self, other: &Self) -> Self {
        Vector {
            tensor: self.tensor.clone() - other.tensor.clone(),
        }
    }
}

/// Allow indexing a vector with usize
impl<T> Index<usize> for Vector<T>
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
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

/// Allow mutable indexing a vector with usize
impl<T> IndexMut<usize> for Vector<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

impl<T> Mul for Vector<T>
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
    fn mul(self, rhs: Self) -> Self::Output {
        self.dot(&rhs)
    }
}


#[cfg(test)]
mod test {
    use super::*;
    // --- Vector Tests ---
    #[test]
    fn test_vector_creation_and_indexing() {
        let v = Vector::from_vec(vec![100, 200, 300]);
        assert_eq!(v.len(), 3);
        assert_eq!(*v.get(1), 200);
    }

    #[test]
    fn test_vector_operations() {
        let v1: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2: Vector<f64> = Vector::from_vec(vec![4.0, 5.0, 6.0]);

        // Test dot product
        assert_eq!(v1.dot(&v2), 32.0); // 1*4 + 2*5 + 3*6 = 32

        // Test element-wise operations
        let sum: Vector<f64> = v1.add(&v2);
        assert_eq!(sum.data(), &[5.0, 7.0, 9.0]);

        let diff: Vector<f64> = v2.sub(&v1);
        assert_eq!(diff.data(), &[3.0, 3.0, 3.0]);
    }
    #[test]
    fn test_vector_from_tensor() {
        let tensor = Tensor::from_vec(vec![3], vec![1, 2, 3]);
        let vec = Vector::from_tensor(tensor);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.data(), &[1, 2, 3]);
    }
    
    #[test]
    fn test_vector_elementwise_addition() {
        let v1 = Vector::from_vec(vec![1, 2, 3]);
        let v2 = Vector::from_vec(vec![4, 5, 6]);
        let v3 = v1.add(&v2);
        // Expected: [5, 7, 9]
        assert_eq!(v3.tensor.data(), &vec![5, 7, 9]);
    }
}