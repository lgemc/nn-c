#include <stdlib.h>
#include <stdalign.h>

#include "mdarray.h"
#include "linear.h"

LinearLayer*     linear_new(MDArray* input, MDArray* labels) {
    // Ensure proper alignment
    LinearLayer* layer = malloc(sizeof(LinearLayer));
    if (!layer) return NULL;

    // Initialize to NULL to avoid issues with cleanup on failure
    layer->weights = NULL;
    layer->biases = NULL;
    layer->input = NULL;
    layer->grad_weights = NULL;
    layer->grad_biases = NULL;

    // Initialize weights with proper shape
    size_t weights_shape[] = {labels->shape[0], input->shape[0]};
    layer->weights = mdarray_create(2, weights_shape, sizeof(double));
    if (!layer->weights) {
        free(layer);
        return NULL;
    }
    mdarray_ones(layer->weights);

    // Initialize biases
    size_t biases_shape[] = {labels->shape[0], 1};
    layer->biases = mdarray_create(2, biases_shape, sizeof(double));
    if (!layer->biases) {
        mdarray_free(layer->weights);
        free(layer);
        return NULL;
    }
    mdarray_zeros(layer->biases);

    // Store input pointer (don't copy)
    layer->input = input;

    return layer;
}

MDArray* linear_forward(LinearLayer* layer, MDArray* input) {
    // Store input for backward pass (just store the pointer, don't copy)
    layer->input = input;

    // Compute output = weights * input
    MDArray* out = mdarray_dot(layer->weights, input);
    if (!out) return NULL;

    // Reshape biases to match the output shape
    size_t new_shape[] = {layer->biases->shape[0], out->shape[1]};
    MDArray* reshaped_biases = mdarray_resize(layer->biases, 2, new_shape);
    if (!reshaped_biases) {
        mdarray_free(out);
        return NULL;
    }

    // Add biases
    MDArray* out_biased = mdarray_sum(out, reshaped_biases);

    // Free intermediate results
    mdarray_free(out);
    mdarray_free(reshaped_biases);

    return out_biased;
}

MDArray* linear_backward(LinearLayer* layer, MDArray* grad_output) {
    // Compute dL/dW = grad_output * input^T
    MDArray* input_transposed = mdarray_transpose(layer->input);
    MDArray* dL_dW = mdarray_dot(grad_output, input_transposed);
    mdarray_free(input_transposed);

    // Compute dL/db = sum of grad_output along the batch dimension
    MDArray* dL_db = mdarray_sum_along_axis(grad_output, 1);

    // Compute dL/dX = W^T * grad_output
    MDArray* weights_transposed = mdarray_transpose(layer->weights);
    MDArray* dL_dX = mdarray_dot(weights_transposed, grad_output);
    mdarray_free(weights_transposed);

    // Store gradients in the layer for later use
    layer->grad_weights = dL_dW;
    layer->grad_biases = dL_db;

    return dL_dX;
}
