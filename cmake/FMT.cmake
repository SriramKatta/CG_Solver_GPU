CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.2.0
)

if(fmt_ADDED)
  get_target_property(FMT_INCLUDES fmt INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(FMT_SYS_INCLUDES fmt INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)

  # keep include dirs, but also tell CMake they are "system"
  set_target_properties(
    fmt PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                   "${FMT_INCLUDES};${FMT_SYS_INCLUDES}"
  )
endif()