use crate::tensor::matrix::Matrix;
use crate::tensor::vector::Vector;
use crate::network::layer::LayerLike;
use std::ops::{Add, Div, Mul, Sub};

pub struct DenseNetwork<T> {
    layers: Vec<Box<dyn LayerLike<T>>>,
}

impl<T> DenseNetwork<T>
where
    T: Copy
        + Default
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialOrd,
{
    // Create a new dense network from a vector of layers
    pub fn new(layers: Vec<Box<dyn LayerLike<T>>>) -> Self {

        let result = DenseNetwork { layers };
        if let Err(str) = result.validate_layers() {
            panic!("{}", str);
        }
        result
    }

    // Append a Layer to the end of the network
    pub fn append_layer(&mut self, layer: Box<dyn LayerLike<T>> ) {
        self.layers.push(layer);
        if let Err(str) = self.validate_layers() {
            panic!("{}", str);
        }
    }

    // Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    // Get immutable reference to layers
    pub fn get_layers(&self) -> &Vec<Box<dyn LayerLike<T>>> {
        &self.layers
    }

    // Get mutable reference to layers
    pub fn get_layers_mut(&mut self) -> &mut Vec<Box<dyn LayerLike<T>>> {
        &mut self.layers
    }
    
    // Get layer at specific index
    pub fn get_layer(&self, index: usize) -> Option<&Box<dyn LayerLike<T>>> {
        self.layers.get(index)
    }
    
    // Get mutable layer at specific index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut Box<dyn LayerLike<T>>> {
        self.layers.get_mut(index)
    }

    // Verify that layer dimensions match
    pub fn validate_layers(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Ok(());
        }
        
        for i in 1..self.layers.len() {
            let prev_layer = &self.layers[i-1];
            let curr_layer = &self.layers[i];
            
            // The number of outputs from previous layer should match
            // number of inputs to current layer
            if prev_layer.output_size() != curr_layer.input_size() {
                return Err(format!(
                    "Layer dimension mismatch: layer {} outputs {} features but layer {} expects {} inputs",
                    i-1, prev_layer.output_size(), i, curr_layer.input_size()
                ));
            }
        }
        Ok(())
    }

    // Get input size required by the network
    pub fn input_size(&self) -> Option<usize> {
        self.layers.first().map(|layer| layer.input_size())
    }
    
    // Get output size produced by the network
    pub fn output_size(&self) -> Option<usize> {
        self.layers.last().map(|layer| layer.output_size())
    }

    // Compute a forward pass for a single input
    pub fn forward(&self, input: Vector<T>) -> Vector<T> {
        let mut current = input;
        for layer in &self.layers {
            current = layer.forward(current);
        }
        current
    }

    // Compute a forward pass for a batch/matrix of inputs
    pub fn forward_batch(&self, inputs: Matrix<T>) -> Matrix<T> {
        let mut current = inputs;
        for layer in &self.layers {
            current = layer.forward_batch(current);
        }
        current
    }
}
