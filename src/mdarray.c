#pragma once

#include "mdarray.h"
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

MDArray* mdarray_create(size_t ndim, size_t* shape, size_t itemsize) {
    MDArray* arr = (MDArray*)malloc(sizeof(MDArray));
    if (!arr) return NULL;

    arr->ndim = ndim;
    arr->itemsize = itemsize;

    // Allocate and copy shape array
    arr->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(arr->shape, shape, ndim * sizeof(size_t));

    // Calculate strides
    arr->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->strides) {
        free(arr->shape);
        free(arr);
        return NULL;
    }

    arr->total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        arr->total_size *= shape[i];
    }

    size_t stride = 1;
    for (size_t i = ndim - 1; i < ndim; i--) {
        arr->strides[i] = stride;
        stride *= shape[i];
    }

    // Allocate data array
    arr->data = malloc(arr->total_size * itemsize);
    if (!arr->data) {
        free(arr->strides);
        free(arr->shape);
        free(arr);
        return NULL;
    }

    arr->owns_data = 1; // This MDArray owns its data

    return arr;
}


void mdarray_free(MDArray* arr) {
    if (arr) {
        if (arr->owns_data)
           free(arr->data);
        free(arr->shape);
        free(arr->strides);
        free(arr);
    }
}

size_t mdarray_calculate_index(MDArray* arr, size_t* indices) {
    size_t flat_index = 0;
    for (size_t i = 0; i < arr->ndim; i++) {
        if (indices[i] >= arr->shape[i]) {  // Add bounds checking
            printf("Index out of bounds: indices[%zu]=%zu >= shape[%zu]=%zu\n", i, indices[i], i, arr->shape[i]);
            return (size_t)-1;  // Return max value to indicate error
        }
        flat_index += indices[i] * arr->strides[i];
    }
    return flat_index;
}

void* mdarray_get_element(MDArray* arr, size_t* indices) {
    size_t index = mdarray_calculate_index(arr, indices);
    if (index == (size_t)-1) {  // Check for the error value
        return NULL;
    }
    return (char*)arr->data + (index * arr->itemsize);
}

void mdarray_set_element(MDArray* arr, size_t* indices, void* value) {
    size_t index = mdarray_calculate_index(arr, indices);
    memcpy((char*)arr->data + (index * arr->itemsize), value, arr->itemsize);
}

// mdarray_dot this is like matmul. Example:
// x       10x768
// y       768x1
// RETURNS 10x1
MDArray* mdarray_dot(MDArray* x, MDArray* y) {
    if(x->ndim != 2 || y->ndim != 2) {
        printf("x and/or y ndim is different than 2\n");
        // TODO: If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly. https://numpy.org/doc/2.1/reference/generated/numpy.matmul.html#numpy.matmul
        return NULL;
    }

    if(x->shape[1] != y->shape[0]) {
        printf("x.shape[1](%zu) different than y.shape[0](%zu)\n", x->shape[1], y->shape[0]);
        return NULL;
    }

    size_t shape[] = {x->shape[0], y->shape[1]};
    MDArray* out = mdarray_create(2, shape, sizeof(double));
    for(size_t i = 0; i < x->shape[0]; i++) {
        for(size_t k = 0; k < y->shape[1]; k++) {
            double outval = 0;  // Reset for each element of output matrix
            for(size_t j = 0; j < y->shape[0]; j++) {
                size_t ix[] = {i, j};
                size_t iy[] = {j, k};
                double xval = *(double*) mdarray_get_element(x, ix);
                double yval = *(double*) mdarray_get_element(y, iy);
                outval += xval*yval;
            }
            size_t idx[] = {i, k};
            mdarray_set_element(out, idx, &outval);  // Write after full sum is computed
        }
    }

    return out;
}


MDArray* mdarray_copy(MDArray* arr, size_t ndim, size_t* start) {
    MDArray* new_arr = (MDArray*)malloc(sizeof(MDArray));
    if (!new_arr || !arr) return NULL;

    new_arr->ndim = arr->ndim - ndim;
    new_arr->itemsize = arr->itemsize;

    // Allocate and copy shape array
    new_arr->shape = (size_t*)malloc(new_arr->ndim * sizeof(size_t));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }

    memcpy(new_arr->shape, &arr->shape[ndim], new_arr->ndim * sizeof(size_t));

    // Calculate strides
    new_arr->strides = (size_t*)malloc(new_arr->ndim * sizeof(size_t));
    if (!new_arr->strides) {
        free(new_arr->shape);
        free(new_arr);
        return NULL;
    }

    new_arr->total_size = 1;
    for (size_t i = 0; i < new_arr->ndim; i++) {
        new_arr->total_size *= new_arr->shape[i];
    }

    size_t stride = 1;
    for (size_t i = new_arr->ndim - 1; i < new_arr->ndim; i--) {
        new_arr->strides[i] = stride;
        stride *= new_arr->shape[i];
    }

    size_t flat_index = 0;
    for (size_t i = 0; i < ndim; i++) {
        flat_index += start[i] * arr->strides[i]; // Find position in old array
    }

    // Start pointer at given index
    new_arr->data = &arr->data[flat_index];
    new_arr->owns_data = 0; // This is a view, so it does not own the data
    if (!new_arr->data) {
        free(new_arr->strides);
        free(new_arr->shape);
        free(new_arr);
        return NULL;
    }

    return new_arr;
}

MDArray* mdarray_resize(MDArray* arr, size_t ndim, size_t* shape) {
    MDArray* new_arr = (MDArray*)malloc(sizeof(MDArray));
    if(!new_arr) return NULL;

    new_arr->ndim = ndim;
    new_arr->itemsize = arr->itemsize;
    new_arr->total_size = arr->total_size;

    // Allocate and copy shape array
    new_arr->shape = (size_t*)malloc(ndim * sizeof(size_t));
    if (!arr->shape) {
        free(arr);
        return NULL;
    }
    memcpy(new_arr->shape, shape, ndim * sizeof(size_t));

    // Calculate strides
    new_arr->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!new_arr->strides) {
        free(new_arr->shape);
        free(new_arr);
        return NULL;
    }

    size_t stride = 1;
    for (size_t i = ndim - 1; i < ndim; i--) {
        new_arr->strides[i] = stride;
        stride *= shape[i];
    }

    new_arr->data = arr->data;
    new_arr->data = arr->data; // or &arr->data[flat_index]
    new_arr->owns_data = 0; // This is a view, so it does not own the data

    return new_arr;
}

MDArray* mdarray_sum(MDArray* a, MDArray* b) {
    if (a->ndim != b->ndim) {
        printf("Arrays must have the same number of dimensions\n");
        return NULL;
    }

    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Arrays must have the same shape\n");
            return NULL;
        }
    }

    MDArray* out = mdarray_create(a->ndim, a->shape, sizeof(double));
    if (!out) return NULL;

    for (size_t i = 0; i < a->total_size; i++) {
        double x = *(double*)((char*)a->data + i * a->itemsize);
        double y = *(double*)((char*)b->data + i * b->itemsize);
        double sum = x + y;
        memcpy((char*)out->data + i * a->itemsize, &sum, a->itemsize);
    }

    return out;
}

MDArray* mdarray_sum_along_axis(MDArray* arr, size_t axis) {
    if (axis >= arr->ndim) {
        printf("Axis out of bounds\n");
        return NULL;
    }

    // Create a new shape for the output array
    size_t* new_shape = (size_t*)malloc((arr->ndim - 1) * sizeof(size_t));
    if (!new_shape) return NULL;

    for (size_t i = 0, j = 0; i < arr->ndim; i++) {
        if (i != axis) {
            new_shape[j++] = arr->shape[i];
        }
    }

    MDArray* result = mdarray_create(arr->ndim - 1, new_shape, arr->itemsize);
    free(new_shape);
    if (!result) return NULL;

    // Initialize the result array with zeros
    mdarray_zeros(result);

    // Sum elements along the specified axis
    size_t* indices = (size_t*)malloc(arr->ndim * sizeof(size_t));
    size_t* result_indices = (size_t*)malloc((arr->ndim - 1) * sizeof(size_t));
    if (!indices || !result_indices) {
        free(indices);
        free(result_indices);
        mdarray_free(result);
        return NULL;
    }

    for (size_t i = 0; i < arr->total_size; i++) {
        size_t temp = i;
        for (size_t j = arr->ndim; j > 0; j--) {
            indices[j - 1] = temp % arr->shape[j - 1];
            temp /= arr->shape[j - 1];
        }

        for (size_t j = 0, k = 0; j < arr->ndim; j++) {
            if (j != axis) {
                result_indices[k++] = indices[j];
            }
        }

        double* result_value = (double*)mdarray_get_element(result, result_indices);
        double* arr_value = (double*)mdarray_get_element(arr, indices);
        *result_value += *arr_value;
    }

    free(indices);
    free(result_indices);

    return result;
}

MDArray* mdarray_transpose(MDArray* arr) {
    if (arr->ndim != 2) {
        printf("Transpose is only implemented for 2D arrays\n");
        return NULL;
    }

    // Create a new array with swapped dimensions
    size_t new_shape[2] = {arr->shape[1], arr->shape[0]};
    MDArray* transposed = mdarray_create(2, new_shape, arr->itemsize);
    if (!transposed) return NULL;

    // Copy elements to the new array with transposed indices
    for (size_t i = 0; i < arr->shape[0]; i++) {
        for (size_t j = 0; j < arr->shape[1]; j++) {
            size_t old_indices[2] = {i, j};
            size_t new_indices[2] = {j, i};
            void* value = mdarray_get_element(arr, old_indices);
            mdarray_set_element(transposed, new_indices, value);
        }
    }

    return transposed;
}

void mdarray_ones(MDArray* arr) {
    for(size_t i = 0; i < arr->total_size; i++) {
        double x = 1;
        memcpy((char*)arr->data + (i * arr->itemsize), &x, arr->itemsize);
    }
}

void mdarray_zeros(MDArray* arr) {
    for(size_t i = 0; i < arr->total_size; i++) {
        double x = 0;
        memcpy((char*)arr->data + (i * arr->itemsize), &x, arr->itemsize);
    }
}