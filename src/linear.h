#pragma once

#include "mdarray.h"

typedef struct {
    MDArray* weights;
    MDArray* biases;
    MDArray* input;
    MDArray* grad_weights;
    MDArray* grad_biases;
    char padding[8];
} LinearLayer;

MDArray* linear_forward(LinearLayer* layer, MDArray* input);
MDArray* linear_backward(LinearLayer* layer, MDArray* grad_output);
LinearLayer* linear_new(MDArray* images, MDArray* labels);

// Structure to hold array metadata
typedef struct {
    MDArray* images;
    MDArray* labels;

    MDArray* weights;
    MDArray* biases;
} LinearModel;

