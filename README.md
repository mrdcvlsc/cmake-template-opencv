# Basic OpenCV CMake Template

## Install Requirements

- For Ubuntu

    ```bash
    sudo apt install libopencv-dev
    ```

- For Windows 

    ```bash
    choco install opencv
    ```

## Build Cmake Template Project

```bash
# configure
cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=FALSE -DCMAKE_BUILD_TYPE=Release

# build
cmake --build build --config Release -j3
```