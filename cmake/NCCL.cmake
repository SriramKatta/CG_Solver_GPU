option(BUILD_NCCL "Build with NCCL support" ON)

if(BUILD_NCCL)

  # Allow user override
  set(NCCL_ROOT "" CACHE PATH "Root directory of NCCL installation")

  # Collect hint paths
  set(NCCL_HINTS
      ${NCCL_ROOT}
      $ENV{NCCL_ROOT}
      $ENV{NCCL_HOME}
      /usr
      /usr/local
      /opt/nccl
      /opt/nvidia/nccl
  )

  find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${NCCL_HINTS}
    PATH_SUFFIXES include
  )

  find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NCCL_HINTS}
    PATH_SUFFIXES lib lib64
  )

  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "Found NCCL:")
    message(STATUS "  Include: ${NCCL_INCLUDE_DIR}")
    message(STATUS "  Library: ${NCCL_LIBRARY}")

    add_library(NCCL::NCCL INTERFACE IMPORTED)
    target_include_directories(NCCL::NCCL INTERFACE ${NCCL_INCLUDE_DIR})
    target_link_libraries(NCCL::NCCL INTERFACE ${NCCL_LIBRARY})

    set(BUILD_NCCL ON)

  else()
    message(WARNING "NCCL not found. Disable NCCL or set NCCL_ROOT/NCCL_HOME.")
    set(BUILD_NCCL OFF)
  endif()

endif()
