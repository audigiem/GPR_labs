# GPR Lab Sessions

This repository contains lab sessions for the Geometric Processing course.

## Structure

- **labNormalEstimation** - Normal estimation and ICP algorithms
- **labReconstruction** - 3D reconstruction from point clouds
- **labCurvatures** - Curvature computation on surfaces
- **labSmoothing** - Mesh smoothing techniques
- **labParametrization** - Surface parameterization methods

## Building

### Build All Labs

To build all lab sessions at once:

```bash
make all
```

### Build Individual Labs

```bash
make lab-normal        # Build Normal Estimation lab
make lab-reconstruction # Build Reconstruction lab
make lab-curvatures    # Build Curvatures lab
make lab-smoothing     # Build Smoothing lab
make lab-parametrization # Build Parametrization lab
```

### Clean Build Files

```bash
make clean             # Clean all build directories
```

## Requirements

- CMake 3.x or higher
- C++ compiler with C++11 support
- OpenGL development libraries
- Eigen library (included)

## Data Files

Data files (meshes, point clouds, scans) are not included in this repository. 
Place your data files in the appropriate directories:
- `labCurvatures/meshes/`
- `labNormalEstimation/scans/`
- `labParametrization/openMeshes/`
- `labReconstruction/points/`
- `labSmoothing/corrupted/`

## Running

After building, executables will be located in each lab's `build/` directory.
Run them from their respective project directories to ensure proper resource loading.
