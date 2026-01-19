
add_executable(test_init_cg test_init_cg.cpp)
target_link_libraries(test_init_cg PRIVATE MPI::MPI_CXX)
target_link_libraries(test_init_cg PRIVATE fmt::fmt)