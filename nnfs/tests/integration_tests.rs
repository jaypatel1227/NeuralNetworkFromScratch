use nnfs::tensor::matrix::Matrix;
use nnfs::tensor::vector::Vector;

#[test]
fn integration_test_matrix_operations() {
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = a.matmul(&b);
    assert_eq!(c.shape(), (2, 2));
}

#[test]
fn integration_test_vector_dot() {
    let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    assert_eq!(v1.dot(&v2), 32.0);
}
