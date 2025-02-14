use crate::tensor;
use crate::tensor::matrix::Matrix;
use crate::tensor::vector::Vector;
use num_traits::Float;
use std::ops::{Add, Div, Mul, Sub};

pub trait LayerLike<T> 
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    fn forward(&self, input: Vector<T>) -> Vector<T>;     
    fn forward_batch(&self, inputs: Matrix<T>) -> Matrix<T>;     
    fn input_size(&self) -> usize;     
    fn output_size(&self) -> usize;     
}


#[derive(Debug, Clone, PartialEq)]
pub struct Layer<T> {
    nodes: Vec<Node<T>>,
    weight_mat: Matrix<T>,
    biases: Vector<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node<T> {
    weights: Vector<T>,
    bias: T,
}

impl<T> Node<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    pub fn new(weights: Vector<T>, bias: T) -> Self {
        Node { weights, bias }
    }

    pub fn forward(&self, input: Vector<T>) -> T {
        assert!(
            input.len() == self.weights.len(),
            "Weigths and input must have the same length for a forward pass."
        );

        self.weights.clone() * input + self.bias
    }
}

impl<T> Layer<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    pub fn new_from_nodes(nodes: Vec<Node<T>>) -> Self {
        let mat_and_vec = Self::weight_mat_and_biases_from_nodes(nodes.clone());
        Layer {
            nodes,
            weight_mat: mat_and_vec.0,
            biases: mat_and_vec.1,
        }
    }

    pub fn new_from_weight_mat_biases(weight_mat: Matrix<T>, biases: Vector<T>) -> Self {
        Layer {
            nodes: Self::nodes_from_weight_mat_and_biases(weight_mat.clone(), biases.clone()),
            weight_mat,
            biases,
        }
    }



    fn weight_mat_and_biases_from_nodes(nodes: Vec<Node<T>>) -> (Matrix<T>, Vector<T>) {
        (
            Matrix::from_vec(
                nodes.len(),
                nodes[0].weights.len(),
                nodes
                    .iter()
                    .flat_map(|node: &Node<T>| node.weights.data().iter().cloned())
                    .collect(),
            ),
            Vector::from_vec(nodes.iter().map(|node: &Node<T>| node.bias).collect()),
        )
    }

    fn nodes_from_weight_mat_and_biases(weight_mat: Matrix<T>, biases: Vector<T>) -> Vec<Node<T>> {
        // assumes each row is a different node
        let shape = weight_mat.shape();
        assert!(shape.0 == biases.len(), "Expected that each row was a node's weights. The shape of the Matrix and Vector doesn't correspond to that.");

        let mut nodes = Vec::<Node<T>>::new();
        for i in 0..biases.len() {
            nodes.push(Node::new(weight_mat.row(i), biases[i]));
        }
        nodes
    }
}

impl<T> LayerLike<T> for Layer<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
 {
    fn forward(&self, input: Vector<T>) -> Vector<T> {
        let result: Vec<T> = self
            .nodes
            .iter()
            .map(|node: &Node<T>| node.forward(input.clone()))
            .collect();
        Vector::from_vec(result)
    }

    fn forward_batch(&self, input: Matrix<T>) -> Matrix<T> {
        Matrix::from_vec(
            input.rows(),
            self.weight_mat.rows(),
            (input.mul(&self.weight_mat.transpose()).into_tensor()
                + self.biases.clone().into_tensor())
            .data()
            .clone(),
        )
    }
    
    fn input_size(&self) -> usize {
        self.weight_mat.cols()
    }

    fn output_size(&self) -> usize {
        self.weight_mat.rows()
    }
}

impl<T> Layer<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd
        + Float
        + From<f64>
        + From<u8>,
{
    pub fn new(input_size: usize, num_nodes: usize) -> Self {
        Layer::new_from_weight_mat_biases(
            tensor::utils::matrix_random(input_size, num_nodes),
            tensor::utils::vector_zeros(num_nodes),
        )
    }
}
