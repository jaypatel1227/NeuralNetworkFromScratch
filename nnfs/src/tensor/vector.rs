use crate::tensor::tensor::Tensor;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

//
// Vector: A 1D tensor wrapper
//

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
    /// Creates a vector from a flat Vec.
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Vector {
            tensor: Tensor::from_vec(vec![len], data),
        }
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.tensor.shape()[0]
    }

    /// Returns the inner tensor struct for this Vector
    pub fn inner_tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Returns the vector
    pub fn data(&self) -> &Vec<T> {
        self.tensor.data()
    }

    /// Returns a reference to the element at index i.
    pub fn get(&self, i: usize) -> &T {
        self.tensor.get(&[i])
    }

    /// Returns a mutable reference to the element at index i.
    pub fn get_mut(&mut self, i: usize) -> &mut T {
        self.tensor.get_mut(&[i])
    }

    /// Dot product between two vectors.
    pub fn dot(&self, other: &Self) -> T {
        assert!(
            self.len() == other.len(),
            "Vectors must have the same length for dot product"
        );
        let mut sum = T::default();
        for i in 0..self.len() {
            sum = sum + (*self.get(i)) * (*other.get(i));
        }
        sum
    }

    /// Element–wise addition.
    pub fn add(&self, other: &Self) -> Self {
        Vector {
            tensor: self.tensor.clone() + other.tensor.clone(),
        }
    }
}

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
    fn test_vector_dot_product() {
        let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        let dot = v1.dot(&v2);
        // Expected: 1*4 + 2*5 + 3*6 = 32
        assert_eq!(dot, 32.0);
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
