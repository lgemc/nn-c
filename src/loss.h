#pragma once

#include "mdarray.h"

// Function to compute Mean Squared Error loss
double mse_loss(MDArray* predictions, MDArray* targets);

// Function to compute the gradient of the Mean Squared Error loss
MDArray* mse_loss_gradient(MDArray* predictions, MDArray* targets);