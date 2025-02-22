#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "unity.h"
#include "mdarray.h"
#include "linear.h"
#include "loss.h"

void test_backpropagation(void) {
    // Create input data (2 samples, 3 features each)
    size_t input_shape[] = {3, 2};
    MDArray* input = mdarray_create(2, input_shape, sizeof(double));
    if (!input) return;

    double input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    memcpy(input->data, input_data, sizeof(input_data));

    // Create target data (2 samples, 1 output each)
    size_t target_shape[] = {1, 2};
    MDArray* targets = mdarray_create(2, target_shape, sizeof(double));
    if (!targets) {
        mdarray_free(input);
        return;
    }

    double target_data[] = {1.0, 2.0};
    memcpy(targets->data, target_data, sizeof(target_data));

    // Initialize linear layer
    LinearLayer* layer = linear_new(input, targets);
    if (!layer) {
        mdarray_free(input);
        mdarray_free(targets);
        return;
    }

    // Initialize weights with small random values
    srand(42); // Fixed seed for reproducibility
    for (size_t i = 0; i < layer->weights->total_size; i++) {
        double* weight = (double*)((char*)layer->weights->data + i * sizeof(double));
        *weight = (((double)rand() / RAND_MAX) * 2 - 1) * 0.01;
    }

    double learning_rate = 0.01;
    int epochs = 50;
    double prev_loss = INFINITY;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        MDArray* predictions = linear_forward(layer, input);
        if (!predictions) break;

        // Compute loss
        double loss = mse_loss(predictions, targets);

        // Early stopping check
        if (loss > prev_loss * 1.5) {
            learning_rate *= 0.5;
        }
        prev_loss = loss;

        printf("Epoch %d, Loss: %f\n", epoch, loss);

        // Backward pass
        MDArray* grad_loss = mse_loss_gradient(predictions, targets);
        if (!grad_loss) {
            mdarray_free(predictions);
            break;
        }

        MDArray* grad_input = linear_backward(layer, grad_loss);

        // Update weights with proper alignment
        for (size_t i = 0; i < layer->weights->total_size; i++) {
            double* weight = (double*)((char*)layer->weights->data + i * sizeof(double));
            double* grad_weight = (double*)((char*)layer->grad_weights->data + i * sizeof(double));
            *weight -= learning_rate * (*grad_weight);
        }

        // Update biases with proper alignment
        for (size_t i = 0; i < layer->biases->total_size; i++) {
            double* bias = (double*)((char*)layer->biases->data + i * sizeof(double));
            double* grad_bias = (double*)((char*)layer->grad_biases->data + i * sizeof(double));
            *bias -= learning_rate * (*grad_bias);
        }

        // Clean up intermediate results
        mdarray_free(predictions);
        mdarray_free(grad_loss);
        if (grad_input) mdarray_free(grad_input);

        // Clear gradients
        if (layer->grad_weights) {
            mdarray_free(layer->grad_weights);
            layer->grad_weights = NULL;
        }
        if (layer->grad_biases) {
            mdarray_free(layer->grad_biases);
            layer->grad_biases = NULL;
        }
    }

    // Clean up
    mdarray_free(layer->weights);
    mdarray_free(layer->biases);
    mdarray_free(input);
    mdarray_free(targets);
}