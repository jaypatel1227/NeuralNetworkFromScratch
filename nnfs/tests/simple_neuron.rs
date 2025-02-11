use nnfs::network::layer::{Layer, Node};
use nnfs::tensor::vector::Vector;

#[test]
fn simple_neuron() {
    let input: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let weights: Vector<f64> = Vector::from_vec(vec![0.2, 0.8, -0.5]);
    let bias: f64 = 2.0;
    let output = (input * weights) + bias;

    assert_eq!(output, 2.3);
}

#[test]
fn simple_layer() {
    let input: Vector<f64> = Vector::from_vec(vec![1.0, 2.0, 3.0, 2.5]);
    let layer = Layer::new_from_nodes(vec![
        Node::new(Vector::from_vec(vec![0.2, 0.8, -0.5, 1.0]), 2.0),
        Node::new(Vector::from_vec(vec![0.5, -0.91, 0.26, -0.5]), 3.0),
        Node::new(Vector::from_vec(vec![-0.26, -0.27, 0.17, 0.87]), 0.5),
    ]);

    assert_eq!(
        layer.forward(input),
        Vector::from_vec(vec![4.8, 1.21, 2.385])
    )
}
