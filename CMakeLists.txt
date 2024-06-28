cmake_minimum_required(VERSION 3.10)
project(micro_nn)

option(WITH_ASAN "Enable AddressSanitizer." OFF)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(WITH_ASAN)
    add_compile_options(-fsanitize=address)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /INCREMENTAL:NO")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /INCREMENTAL:NO")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /INCREMENTAL:NO")
        # Attempt to find the ASan runtime DLL by looking for one of the possible names it could have.
        find_file(ASAN_RUNTIME_DLL NAMES clang_rt.asan_dbg_dynamic-x86_64.dll clang_rt.asan_dynamic-x86_64.dll PATHS ENV PATH NO_DEFAULT_PATH)
        if(NOT ASAN_RUNTIME_DLL)
            message(WARNING "WITH_ASAN is ON, but clang_rt.asan_*.dll was not found on PATH. Add the location to PATH, e.g. C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\<version>\\bin\\Hostx64\\x64")
        endif()
    else()
        add_link_options(-fsanitize=address)        
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fconstexpr-ops-limit=100000000)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fconstexpr-steps=100000000)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/constexpr:steps100000000)
endif()

add_subdirectory(src)
add_subdirectory(test)

enable_testing()