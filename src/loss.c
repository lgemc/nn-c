#include <math.h>
#include <stdio.h>
#include <string.h>

#include "loss.h"

double mse_loss(MDArray* predictions, MDArray* targets) {
    if (predictions->total_size != targets->total_size) {
        printf("Predictions and targets must have the same size\n");
        return -1.0;
    }

    double loss = 0.0;
    for (size_t i = 0; i < predictions->total_size; i++) {
        double pred = *(double*)((char*)predictions->data + i * predictions->itemsize);
        double target = *(double*)((char*)targets->data + i * targets->itemsize);
        loss += pow(pred - target, 2);
    }

    return loss / predictions->total_size;
}

MDArray* mse_loss_gradient(MDArray* predictions, MDArray* targets) {
    if (predictions->total_size != targets->total_size) {
        printf("Predictions and targets must have the same size\n");
        return NULL;
    }

    MDArray* grad = mdarray_create(predictions->ndim, predictions->shape, sizeof(double));
    if (!grad) return NULL;

    for (size_t i = 0; i < predictions->total_size; i++) {
        double pred = *(double*)((char*)predictions->data + i * predictions->itemsize);
        double target = *(double*)((char*)targets->data + i * targets->itemsize);
        double gradient = 2 * (pred - target) / predictions->total_size;
        memcpy((char*)grad->data + i * grad->itemsize, &gradient, grad->itemsize);
    }

    return grad;
}