#include "unity.h"

// Declarations of test functions from test_mdarray.c
void test_mdarray_creation_and_access(void);
void test_mdarray_dot_product(void);
void test_md_array_sum_one_dimension(void);
void test_md_array_sum_three_dimensions(void);

// Declarations of test functions from test_linear.c
void test_backpropagation(void);

int main(void) {
    UNITY_BEGIN();

    // Run tests from test_mdarray.c
    RUN_TEST(test_mdarray_creation_and_access);
    RUN_TEST(test_mdarray_dot_product);
    RUN_TEST(test_md_array_sum_one_dimension);
    RUN_TEST(test_md_array_sum_three_dimensions);

    // Run tests from test_linear.c
    RUN_TEST(test_backpropagation);

    return UNITY_END();
}