mod tensor;
use tensor::matrix::Matrix;
use tensor::tensor::Tensor;
use tensor::vector::Vector;

//
// Example Usage
//

fn main() {
    // --- Tensor Element–wise Operations with Broadcasting ---
    let t1 = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    // Create a "bias" vector (shape [2]) that will be broadcasted along the first axis.
    let bias = Tensor::from_vec(vec![2], vec![10.0, 20.0]);
    // Adding bias to each row of t1 (broadcast: [2,2] and [1,2] broadcast to [2,2])
    let t_add = t1.clone() + bias.clone();
    println!("Tensor t1: {:?}", t1);
    println!("Bias: {:?}", bias);
    println!("t1 + bias: {:?}", t_add);

    // --- Matrix Multiplication ---
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = a.matmul(&b);
    println!("Matrix a: {:?}", a);
    println!("Matrix b: {:?}", b);
    println!("a.matmul(b): {:?}", c);

    // --- Transpose and Reshape ---
    let a_trans = a.transpose();
    println!("Transpose of a: {:?}", a_trans);
    let a_reshaped = a.inner_tensor().reshape(vec![3, 2]);
    println!("Reshaped a (3x2): {:?}", a_reshaped);

    // --- Reduction Operations ---
    // Sum along axis 0 (summing over rows yields a 1xN tensor)
    let sum0 = a.inner_tensor().sum(0);
    println!("Sum of a along axis 0: {:?}", sum0);
    // Mean along axis 1 (requires T: From<f64>)
    let mean1 = a.inner_tensor().mean(1);
    println!("Mean of a along axis 1: {:?}", mean1);
    let max0 = a.inner_tensor().max(0);
    println!("Max of a along axis 0: {:?}", max0);
    let min1 = a.inner_tensor().min(1);
    println!("Min of a along axis 1: {:?}", min1);

    // --- Vector Operations ---
    let v1 = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::from_vec(vec![4.0, 5.0, 6.0]);
    println!("Vector v1: {:?}", v1);
    println!("Vector v2: {:?}", v2);
    println!("Dot product (v1 * v2): {}", v1.clone() * v2.clone());
    let v_add = v1.add(&v2);
    println!("v1 + v2: {:?}", v_add);
}
