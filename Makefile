.PHONY: all clean lab-normal lab-reconstruction lab-curvatures lab-smoothing lab-parametrization

# Build all labs
all: lab-normal lab-reconstruction lab-curvatures lab-smoothing lab-parametrization

# Normal Estimation Lab
lab-normal:
	@echo "=== Building Normal Estimation Lab ==="
	@mkdir -p labNormalEstimation/01-icp-base/build
	@cd labNormalEstimation/01-icp-base/build && cmake .. && make
	@echo "✓ Normal Estimation Lab built successfully\n"

# Reconstruction Lab
lab-reconstruction:
	@echo "=== Building Reconstruction Lab ==="
	@mkdir -p labReconstruction/02-reconstruction-base/build
	@cd labReconstruction/02-reconstruction-base/build && cmake .. && make
	@echo "✓ Reconstruction Lab built successfully\n"

# Curvatures Lab
lab-curvatures:
	@echo "=== Building Curvatures Lab ==="
	@mkdir -p labCurvatures/03-curvatures-base/build
	@cd labCurvatures/03-curvatures-base/build && cmake .. && make
	@echo "✓ Curvatures Lab built successfully\n"

# Smoothing Lab
lab-smoothing:
	@echo "=== Building Smoothing Lab ==="
	@mkdir -p labSmoothing/04-smoothing-base/build
	@cd labSmoothing/04-smoothing-base/build && cmake .. && make
	@echo "✓ Smoothing Lab built successfully\n"

# Parametrization Lab
lab-parametrization:
	@echo "=== Building Parametrization Lab ==="
	@mkdir -p labParametrization/06-parameterization-base/build
	@cd labParametrization/06-parameterization-base/build && cmake .. && make
	@echo "✓ Parametrization Lab built successfully\n"

# Clean all build directories
clean:
	@echo "=== Cleaning all build directories ==="
	@rm -rf labNormalEstimation/01-icp-base/build
	@rm -rf labReconstruction/02-reconstruction-base/build
	@rm -rf labCurvatures/03-curvatures-base/build
	@rm -rf labSmoothing/04-smoothing-base/build
	@rm -rf labParametrization/06-parameterization-base/build
	@echo "✓ All build directories cleaned\n"
