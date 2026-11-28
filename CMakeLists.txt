# Builds three things from one tree:
#   N-body          interactive OpenGL viewer (original project, unchanged)     NBODY_GUI
#   nbody_core      simulation library without OpenGL + nbody-headless/compare  NBODY_HEADLESS
#   nbody-mpi       distributed binary (step 2); CUDA kernel in step 6          NBODY_MPI / NBODY_CUDA
# Headless build: cmake -S . -B build-headless -DNBODY_GUI=OFF
cmake_minimum_required(VERSION 3.18)
project(N-body LANGUAGES C CXX) # C is needed for glad

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NBODY_GUI      "Build the interactive OpenGL viewer"                  ON)
option(NBODY_HEADLESS "Build nbody_core, nbody-headless and nbody-compare"   ON)
option(NBODY_MPI      "Build the distributed nbody-mpi binary (step 2)"      OFF)
option(NBODY_CUDA     "Build the CUDA force kernel (step 6)"                 OFF)

# -march=native lets an image built on one machine use instructions another machine
# lacks ("illegal instruction"). Docker builds pass -DNBODY_ARCH=x86-64-v3.
set(NBODY_ARCH "native" CACHE STRING "-march value for GCC/Clang")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Optimization and warning flags, C++ files only so nvcc never sees -march
function(nbody_compile_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:fast> /W4 /EHsc)
  else()
    set(rel_cxx "$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>")
    target_compile_options(${target} PRIVATE
      $<${rel_cxx}:-O3>
      $<${rel_cxx}:-march=${NBODY_ARCH}>
      $<${rel_cxx}:-ffast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
  endif()
endfunction()

#
# Interactive OpenGL viewer (src/ and include/, unchanged)
#
if(NBODY_GUI)
  include(FetchContent)
  set(FETCHCONTENT_QUIET FALSE)
  set(GLFW_BUILD_DOCS FALSE)
  set(GLFW_BUILD_EXAMPLES FALSE)
  set(GLFW_BUILD_TESTS FALSE)

  FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.4
    GIT_PROGRESS TRUE)
  FetchContent_Declare(glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad
    GIT_TAG v0.1.34
    GIT_PROGRESS TRUE)
  FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm
    GIT_TAG 0.9.9.8
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(glfw glad glm)

  file(GLOB PROJECT_SOURCES_CPP "src/*.cpp" "src/*/*.cpp" "src/*/*/*.cpp")
  add_executable(${PROJECT_NAME} ${PROJECT_SOURCES_CPP})
  nbody_compile_options(${PROJECT_NAME})
  if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /openmp:llvm)
  endif()
  target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
  target_link_libraries(${PROJECT_NAME} PRIVATE glfw glad glm OpenMP::OpenMP_CXX)
endif()

#
# Headless core, no OpenGL/GLFW/glm
#
if(NBODY_HEADLESS OR NBODY_MPI)
  add_library(nbody_core STATIC
    core/src/forces.cpp
    core/src/forces_sequential.cpp
    core/src/forces_openmp.cpp
    core/src/integrator.cpp
    core/src/initial_conditions.cpp
    core/src/checkpoint.cpp)
  target_include_directories(nbody_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
  target_link_libraries(nbody_core PUBLIC OpenMP::OpenMP_CXX)
  nbody_compile_options(nbody_core)

  # Keeps the sequential kernel scalar so it's an honest bottom rung of the speedup ladder
  if(NOT MSVC)
    set_source_files_properties(core/src/forces_sequential.cpp PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize")
  endif()

  if(NBODY_CUDA)
    # core/src/forces_cuda.cu is added in step 6
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 3050 Ti (Ampere)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(nbody_core PRIVATE core/src/forces_cuda.cu)
    target_compile_definitions(nbody_core PUBLIC NBODY_HAVE_CUDA)
    target_link_libraries(nbody_core PUBLIC CUDA::cudart)
  endif()

  add_executable(nbody-headless apps/nbody_headless.cpp)
  target_link_libraries(nbody-headless PRIVATE nbody_core)
  nbody_compile_options(nbody-headless)

  add_executable(nbody-compare apps/nbody_compare.cpp)
  target_link_libraries(nbody-compare PRIVATE nbody_core)
  nbody_compile_options(nbody-compare)
endif()

if(NBODY_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  # nbody-mpi (apps/nbody_mpi.cpp) is added in step 2
endif()
