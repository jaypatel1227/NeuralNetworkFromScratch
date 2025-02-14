use nnfs::network::layer::{Layer, LayerLike, Node};
use nnfs::tensor::matrix::Matrix;
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

#[test]
fn simple_batch_layer() {
    let inputs = Matrix::from_vec(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8,
        ],
    );

    let weights = Matrix::from_vec(
        3,
        4,
        vec![
            0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87,
        ],
    );

    let biases = Vector::from_vec(vec![2.0, 3.0, 0.5]);
    let layer = Layer::new_from_weight_mat_biases(weights, biases);

    assert_eq!(
        layer.forward_batch(inputs),
        Matrix::from_vec(
            3,
            3,
            vec![
                4.8,
                1.21,
                2.385,
                8.9,
                -1.8099999999999996,
                0.19999999999999996,
                1.4100000000000001,
                1.0509999999999997,
                0.025999999999999912
            ]
        )
    )
}

#[test]
fn two_layer_batch_forward() {
    let inputs = Matrix::from_vec(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8,
        ],
    );

    let weights1 = Matrix::from_vec(
        3,
        4,
        vec![
            0.2, 0.8, -0.5, 1.0, 0.5, -0.91, 0.26, -0.5, -0.26, -0.27, 0.17, 0.87,
        ],
    );

    let biases1 = Vector::from_vec(vec![2.0, 3.0, 0.5]);

    let weights2 = Matrix::from_vec(
        3,
        3,
        vec![0.1, -0.14, 0.5, -0.5, 0.12, -0.33, -0.44, 0.73, -0.13],
    );

    let biases2 = Vector::from_vec(vec![-1.0, 2.0, -0.5]);

    let layer1 = Layer::new_from_weight_mat_biases(weights1, biases1);
    let layer2 = Layer::new_from_weight_mat_biases(weights2, biases2);

    let layer1_outputs = layer1.forward_batch(inputs);
    let layer2_outputs = layer2.forward_batch(layer1_outputs);

    assert_eq!(
        layer2_outputs,
        Matrix::from_vec(
            3,
            3,
            vec![
                0.5030999999999999,
                -1.0418499999999997,
                -2.0387500000000003,
                0.24340000000000028,
                -2.7332,
                -5.7633,
                -0.99314,
                1.41254,
                -0.3565500000000003
            ]
        )
    );
}
