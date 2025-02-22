#pragma once
#include "mdarray.h"

typedef struct Layer {
  void* layer_data;
    MDArray* (*forward)(struct Layer*, MDArray*);
    // Backward function should return the gradient with respect to the input
    MDArray* (*backward)(struct Layer*, MDArray*);
} Layer;

Layer* layer_create(void* layer_data, MDArray* (*forward)(Layer*, MDArray*), MDArray* (*backward)(Layer*, MDArray*));
void layer_free(Layer* layer);