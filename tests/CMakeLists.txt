add_executable(tests
        unity/src/unity.c
        test_mdarray.c
        test_linear.c
        test_runner.c
        ${CMAKE_SOURCE_DIR}/src/mdarray.c
        ${CMAKE_SOURCE_DIR}/src/linear.c
        ${CMAKE_SOURCE_DIR}/src/loss.c
)

# Include Unity headers
target_include_directories(tests
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unity/src)
target_link_libraries(tests m)
# Enable testing
enable_testing()
add_test(NAME RunUnitTests COMMAND tests)