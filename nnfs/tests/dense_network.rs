use nnfs::network::dense_network::DenseNetwork;
use nnfs::network::layer::{Layer, LayerLike, Node};
use nnfs::tensor::matrix::Matrix;
use nnfs::tensor::vector::Vector;

#[test]
fn test_empty_network() {
    let mut network: DenseNetwork<f64> = DenseNetwork::new(vec![]);
    assert_eq!(network.get_layers_mut().len(), 0);
}

#[test]
fn test_append_layer() {
    let mut network: DenseNetwork<f64> = DenseNetwork::new(vec![]);
    let layer = Layer::new_from_nodes(vec![
        Node::new(Vector::from_vec(vec![0.2, 0.8]), 2.0),
        Node::new(Vector::from_vec(vec![0.5, -0.91]), 3.0),
    ]);
    network.append_layer(Box::new(layer.clone()));
    assert_eq!(network.get_layers_mut().len(), 1);
    
    // Test layer behavior instead of direct comparison
    let input = Vector::from_vec(vec![1.0, 2.0]);
    assert_eq!(network.get_layers_mut()[0].forward(input.clone()), layer.forward(input));
}

#[test]
fn test_single_layer_forward() {
    let layer = Layer::new_from_nodes(vec![
        Node::new(Vector::from_vec(vec![0.2, 0.8]), 2.0),
        Node::new(Vector::from_vec(vec![0.5, -0.91]), 3.0),
    ]);
    let network = DenseNetwork::new(vec![Box::new(layer)]);
    
    let input = Vector::from_vec(vec![1.0, 2.0]);
    let output = network.forward(input);
    
    assert_eq!(output, Vector::from_vec(vec![3.8, 1.68]));
}

#[test]
fn test_two_layer_forward() {
    let layer1 = Layer::new_from_nodes(vec![
        Node::new(Vector::from_vec(vec![0.2, 0.8]), 2.0),
        Node::new(Vector::from_vec(vec![0.5, -0.91]), 3.0),
    ]);
    
    let layer2 = Layer::new_from_nodes(vec![
        Node::new(Vector::from_vec(vec![0.1, -0.5]), -1.0),
        Node::new(Vector::from_vec(vec![-0.14, 0.12]), 2.0),
    ]);
    
    let network = DenseNetwork::new(vec![Box::new(layer1), Box::new(layer2)]);
    let input = Vector::from_vec(vec![1.0, 2.0]);
    let output = network.forward(input);
    
    // Values can be calculated by hand: first layer output [3.8, 1.68], then through second layer
    let expected = Vector::from_vec(vec![-1.46, 1.6696]);
    assert_eq!(output, expected);
}

#[test]
fn test_batch_forward() {
    let layer = Layer::new_from_nodes(vec![
        Node::new(Vector::from_vec(vec![0.2, 0.8]), 2.0),
        Node::new(Vector::from_vec(vec![0.5, -0.91]), 3.0),
    ]);
    let network = DenseNetwork::new(vec![Box::new(layer)]);
    
    let inputs = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let outputs = network.forward_batch(inputs);
    
    assert_eq!(
        outputs,
        Matrix::from_vec(2, 2, vec![3.8, 1.68, 5.800000000000001, 0.8599999999999999])
    );
} 