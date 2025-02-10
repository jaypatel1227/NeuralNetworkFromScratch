use nnfs::tensor::vector::Vector;

#[test]
fn simple_neuron() {
    let input: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let weights: Vector<f64> = Vector::from_vec(vec![0.2, 0.8, -0.5]);
    let bias: f64 = 2.0;
    let output = (input * weights) + bias;

    assert_eq!(output, 2.3);
}
