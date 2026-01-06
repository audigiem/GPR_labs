#include <iostream>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "RBFFunction.h"
#include "TrackedBiCGSTAB.h"

#ifdef _OPENMP
#include <omp.h>
#endif


/* Initialize everything to be able to compute the implicit distance to the reconstructed
   point cloud at arbitrary points that are close enough to the point cloud. As should be
   obvious by the name of the class, the distance has to be computed using RBFs.
 */

void RBFFunction::createConstraintPoints(const std::vector<glm::vec3> &points,
                                          const std::vector<glm::vec3> &normals,
                                          float offset)
{
	unsigned int n = points.size();

	constraintPoints.clear();
	constraintValues.clear();

	// Add constraint points: original + positive/negative offset
	for (unsigned int i = 0; i < n; i++)
	{
		// Original point: f(pi) = 0
		constraintPoints.push_back(points[i]);
		constraintValues.push_back(0.0f);

		// Positive offset: f(pi + d*ni) = d
		constraintPoints.push_back(points[i] + offset * normals[i]);
		constraintValues.push_back(offset);

		// Negative offset: f(pi - d*ni) = -d
		constraintPoints.push_back(points[i] - offset * normals[i]);
		constraintValues.push_back(-offset);
	}

	std::cout << "\n========================================" << std::endl;
	std::cout << "RBF RECONSTRUCTION" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Original points:     " << n << std::endl;
	std::cout << "Constraint points:   " << constraintPoints.size() << " (3x augmentation)" << std::endl;
}

float RBFFunction::estimateAndAdaptSearchRadius(float initialRadius, bool isLargeSystem, unsigned int m)
{
	float searchRadius = initialRadius;

	if (isLargeSystem) {
		// For LARGE systems: aggressive sparsity control
		std::cout << "System size:         LARGE (" << m << " equations)" << std::endl;
		std::cout << "Strategy:            Adaptive sparsity control" << std::endl;

		int targetMaxNeighbors = 80;  // Stricter for large systems

		// Quick test: sample a few points to estimate neighbor count
		std::cout << "\nAnalyzing matrix density..." << std::endl;
		int sampleSize = std::min(100, (int)m);
		int totalNeighbors = 0;
		vector<pair<size_t, float>> testNeighbors;

		for (int sample = 0; sample < sampleSize; sample++) {
			int idx = (m * sample) / sampleSize;
			testNeighbors.clear();
			knn.getNeighborsInRadius(constraintPoints[idx], searchRadius, testNeighbors);
			totalNeighbors += testNeighbors.size();
		}

		int avgNeighbors = totalNeighbors / sampleSize;
		std::cout << "  Initial avg neighbors: " << avgNeighbors << " (radius: " << searchRadius << ")" << std::endl;

		// Adaptively reduce search radius if too many neighbors
		if (avgNeighbors > targetMaxNeighbors) {
			float reductionFactor = sqrt((float)targetMaxNeighbors / avgNeighbors);
			searchRadius *= reductionFactor;
			std::cout << "  → Reducing radius to " << searchRadius << " (" << int(reductionFactor * 100) << "%)" << std::endl;

			// Re-estimate
			totalNeighbors = 0;
			for (int sample = 0; sample < sampleSize; sample++) {
				int idx = (m * sample) / sampleSize;
				testNeighbors.clear();
				knn.getNeighborsInRadius(constraintPoints[idx], searchRadius, testNeighbors);
				totalNeighbors += testNeighbors.size();
			}
			avgNeighbors = totalNeighbors / sampleSize;
			std::cout << "  → Adapted avg neighbors: " << avgNeighbors << std::endl;
		}
	} else {
		// For SMALL systems: use original approach (works well)
		std::cout << "System size:         SMALL (" << m << " equations)" << std::endl;
		std::cout << "Strategy:            Standard compact support (3sig)" << std::endl;
	}

	return searchRadius;
}

void RBFFunction::buildMatrixParallel(Eigen::SparseMatrix<double> &A,
                                       Eigen::VectorXd &b,
                                       float searchRadius,
                                       unsigned int m)
{
#if defined(_OPENMP) && defined(USE_PARALLEL_RBF)
	std::cout << "\n========================================" << std::endl;
	std::cout << "PHASE 1: Matrix Construction" << std::endl;
	std::cout << "========================================" << std::endl;
	int numThreads = omp_get_max_threads();
	std::cout << "Mode:                Parallel (" << numThreads << " threads)" << std::endl;

	auto matrixStartTime = std::chrono::high_resolution_clock::now();

	// Create thread-local triplet vectors
	std::vector<std::vector<Eigen::Triplet<double>>> localTriplets(numThreads);

	// Pre-estimate triplets per thread
	int avgNeighbors = 100;
	int rowsPerThread = (m + numThreads - 1) / numThreads;
	for (int t = 0; t < numThreads; t++) {
		localTriplets[t].reserve(rowsPerThread * avgNeighbors);
	}

	// Build matrix A and vector b (PARALLELIZED)
	#pragma omp parallel
	{
		int threadId = omp_get_thread_num();
		vector<pair<size_t, float>> neighbors;

		#pragma omp for schedule(dynamic, 100)
		for (unsigned int i = 0; i < m; i++)
		{
			b(i) = constraintValues[i];

			// Use KNN to find only nearby points
			neighbors.clear();
			knn.getNeighborsInRadius(constraintPoints[i], searchRadius, neighbors);

			// Compute phi only for neighbors
			for (const auto &neighbor : neighbors)
			{
				unsigned int j = neighbor.first;
				float distSquared = neighbor.second;
				float r = sqrt(distSquared);
				float phi = evaluatePhi(r);

				if (phi != 0.0f)
				{
					localTriplets[threadId].push_back(Eigen::Triplet<double>(i, j, phi));
				}
			}
		}
	}

	// Merge all thread-local triplets
	std::cout << "Building...          " << std::flush;
	size_t totalTriplets = 0;
	for (int t = 0; t < numThreads; t++) {
		totalTriplets += localTriplets[t].size();
	}

	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(totalTriplets);
	for (int t = 0; t < numThreads; t++) {
		triplets.insert(triplets.end(), localTriplets[t].begin(), localTriplets[t].end());
	}

	A.setFromTriplets(triplets.begin(), triplets.end());

	auto matrixEndTime = std::chrono::high_resolution_clock::now();
	auto matrixDuration = std::chrono::duration_cast<std::chrono::milliseconds>(matrixEndTime - matrixStartTime);

	std::cout << "Done!" << std::endl;
	std::cout << "Non-zero entries:    " << triplets.size() << " (" << (100.0 * triplets.size() / (m * (double)m)) << "% sparse)" << std::endl;
	std::cout << "Construction time:   " << matrixDuration.count() / 1000.0 << "s" << std::endl;
#endif
}

void RBFFunction::buildMatrixSequential(Eigen::SparseMatrix<double> &A,
                                         Eigen::VectorXd &b,
                                         float searchRadius,
                                         unsigned int m)
{
	std::cout << "\n========================================" << std::endl;
	std::cout << "PHASE 1: Matrix Construction" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Mode:                Sequential" << std::endl;
	auto matrixStartTime = std::chrono::high_resolution_clock::now();

	std::vector<Eigen::Triplet<double>> triplets;
	vector<pair<size_t, float>> neighbors;

	// Build matrix A and vector b with progress
	std::cout << "Progress:            " << std::flush;
	for (unsigned int i = 0; i < m; i++)
	{
		b(i) = constraintValues[i];

		if (i % (m / 10) == 0)
		{
			std::cout << int(100.0 * i / m) << "% " << std::flush;
		}

		// Use KNN to find only nearby points
		neighbors.clear();
		knn.getNeighborsInRadius(constraintPoints[i], searchRadius, neighbors);

		// Compute phi only for neighbors
		for (const auto &neighbor : neighbors)
		{
			unsigned int j = neighbor.first;
			float distSquared = neighbor.second;
			float r = sqrt(distSquared);
			float phi = evaluatePhi(r);

			if (phi != 0.0f)
			{
				triplets.push_back(Eigen::Triplet<double>(i, j, phi));
			}
		}
	}

	A.setFromTriplets(triplets.begin(), triplets.end());

	std::cout << "100%" << std::endl;
	
	auto matrixEndTime = std::chrono::high_resolution_clock::now();
	auto matrixDuration = std::chrono::duration_cast<std::chrono::milliseconds>(matrixEndTime - matrixStartTime);

	std::cout << "Non-zero entries:    " << triplets.size() << " (" << (100.0 * triplets.size() / (m * (double)m)) << "% sparse)" << std::endl;
	std::cout << "Construction time:   " << matrixDuration.count() / 1000.0 << "s" << std::endl;
}

double RBFFunction::computeAdaptiveRegularization(const Eigen::SparseMatrix<double> &A,
                                                   unsigned int m,
                                                   bool isLargeSystem)
{
	// ADAPTIVE REGULARIZATION based on system size and matrix density
	size_t nonZeros = A.nonZeros();
	double avgNonZerosPerRow = (double)nonZeros / m;

	std::cout << "\n========================================" << std::endl;
	std::cout << "PHASE 2: Regularization" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Avg neighbors/row:   " << int(avgNonZerosPerRow) << std::endl;

	double lambda;

	if (isLargeSystem) {
		// Large or dense systems need stronger regularization
		// Matrix becomes ill-conditioned as size or density grows

		if (avgNonZerosPerRow > 120) {
			lambda = 100.0;
			std::cout << "Strategy:            λ = " << lambda << " (very strong - extreme density)" << std::endl;
		}
		else if (avgNonZerosPerRow > 100) {
			lambda = 50.0;
			std::cout << "Strategy:            λ = " << lambda << " (strong - high density)" << std::endl;
		}
		else if (avgNonZerosPerRow > 80) {
			lambda = 20.0;
			std::cout << "Strategy:            λ = " << lambda << " (moderate-strong)" << std::endl;
		}
		else {
			lambda = 5.0;
			std::cout << "Strategy:            λ = " << lambda << " (moderate)" << std::endl;
		}
	}
	else {
		// Small, sparse systems: use light regularization
		lambda = 0.1;
		std::cout << "Strategy:            λ = " << lambda << " (light - well-conditioned)" << std::endl;
	}

	return lambda;
}

void RBFFunction::applyRegularization(Eigen::SparseMatrix<double> &A,
                                       double lambda,
                                       unsigned int m)
{
	auto regStartTime = std::chrono::high_resolution_clock::now();

	// Apply regularization: A' = A + λI
	for (unsigned int i = 0; i < m; i++)
	{
		A.coeffRef(i, i) += lambda;
	}
	
	auto regEndTime = std::chrono::high_resolution_clock::now();
	auto regDuration = std::chrono::duration_cast<std::chrono::milliseconds>(regEndTime - regStartTime);
	std::cout << "Application time:    " << regDuration.count() / 1000.0 << "s" << std::endl;
}

Eigen::VectorXd RBFFunction::solveLargeSystem(const Eigen::SparseMatrix<double> &A,
                                               const Eigen::VectorXd &b,
                                               unsigned int m)
{
	// LARGE SYSTEM: Use IncompleteLUT preconditioner with relaxed tolerance
	std::cout << "\n========================================" << std::endl;
	std::cout << "PHASE 3: Solving Linear System" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Solver:              BiCGSTAB" << std::endl;
	std::cout << "Preconditioner:      IncompleteLUT" << std::endl;
	std::cout << "Computing preconditioner..." << std::flush;

	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
	solver.preconditioner().setFillfactor(2);
	solver.preconditioner().setDroptol(0.01);
	solver.setMaxIterations(2000);  // Fewer iterations for large systems
	solver.setTolerance(1e-3);  // More relaxed tolerance

	auto precondStartTime = std::chrono::high_resolution_clock::now();
	solver.compute(A);
	auto precondEndTime = std::chrono::high_resolution_clock::now();
	auto precondDuration = std::chrono::duration_cast<std::chrono::milliseconds>(precondEndTime - precondStartTime);

	std::cout << " Done! (" << precondDuration.count() / 1000.0 << "s)" << std::endl;

	if (solver.info() != Eigen::Success)
	{
		std::cerr << "ERROR: Preconditioner computation failed!" << std::endl;
		return Eigen::VectorXd();
	}

	std::cout << "Starting BiCGSTAB iterations with tracking..." << std::endl;
	auto solverStartTime = std::chrono::high_resolution_clock::now();

	Eigen::VectorXd x = CustomSolvers::solveWithTracking(solver, A, b, 100);

	auto solverEndTime = std::chrono::high_resolution_clock::now();
	auto solverDuration = std::chrono::duration_cast<std::chrono::milliseconds>(solverEndTime - solverStartTime);

	Eigen::VectorXd residual = b - A * x;
	double relativeResidual = residual.norm() / b.norm();
	int iterations = solver.iterations();

	std::cout << "Done!" << std::endl;
	std::cout << "Iterations:          " << iterations;
	if (iterations <= 5) {
		std::cout << " EXCELLENT!";
	}
	std::cout << std::endl;
	std::cout << "Relative residual:   " << relativeResidual;
	if (relativeResidual < 0.001) {
		std::cout << " ✓ Excellent";
	} else if (relativeResidual < 0.01) {
		std::cout << " ✓ Good";
	} else if (relativeResidual > 0.5) {
		std::cout << " Poor convergence";
	}
	std::cout << std::endl;
	std::cout << "Solver time:         " << solverDuration.count() / 1000.0 << "s" << std::endl;

	return x;
}

Eigen::VectorXd RBFFunction::solveSmallSystem(const Eigen::SparseMatrix<double> &A,
                                               const Eigen::VectorXd &b,
                                               unsigned int m)
{
	// SMALL SYSTEM: Use simple diagonal preconditioner (original working code)
	std::cout << "\n========================================" << std::endl;
	std::cout << "PHASE 3: Solving Linear System" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Solver:              BiCGSTAB" << std::endl;
	std::cout << "Preconditioner:      Diagonal" << std::endl;

	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
	solver.setMaxIterations(5000);
	solver.setTolerance(1e-4);
	solver.compute(A);

	if (solver.info() != Eigen::Success)
	{
		std::cerr << "ERROR: Decomposition failed!" << std::endl;
		return Eigen::VectorXd();
	}

	std::cout << "Starting BiCGSTAB iterations with tracking..." << std::endl;
	auto solverStartTime = std::chrono::high_resolution_clock::now();

	Eigen::VectorXd x = CustomSolvers::solveWithTracking(solver, A, b, 100);

	auto solverEndTime = std::chrono::high_resolution_clock::now();
	auto solverDuration = std::chrono::duration_cast<std::chrono::milliseconds>(solverEndTime - solverStartTime);

	Eigen::VectorXd residual = b - A * x;
	double relativeResidual = residual.norm() / b.norm();
	int iterations = solver.iterations();

	std::cout << "Done!" << std::endl;
	std::cout << "Iterations:          " << iterations;
	if (iterations <= 5) {
		std::cout << " EXCELLENT!";
	}
	std::cout << std::endl;
	std::cout << "Relative residual:   " << relativeResidual;
	if (relativeResidual < 0.0001) {
		std::cout << " ✓ Excellent";
	} else if (relativeResidual < 0.01) {
		std::cout << " ✓ Very good";
	} else if (relativeResidual < 0.1) {
		std::cout << " ✓ Good";
	} else {
		std::cout << " Poor";
	}
	std::cout << std::endl;
	std::cout << "Solver time:         " << solverDuration.count() / 1000.0 << "s" << std::endl;

	return x;
}

void RBFFunction::storeCoefficients(const Eigen::VectorXd &x, unsigned int m)
{
	coefficients.resize(m);
	for (unsigned int i = 0; i < m; i++)
	{
		coefficients[i] = x(i);
	}
}

void RBFFunction::init(const PointCloud *pointCloud, float standardDeviation, float supportRadius, bool useParallel)
{
	auto totalStartTime = std::chrono::high_resolution_clock::now();
	
	// Step 1: Initialize basic parameters
	cloud = pointCloud;
	c = standardDeviation;
	radius = supportRadius;

	const vector<glm::vec3> &points = cloud->getPoints();
	const vector<glm::vec3> &normals = cloud->getNormals();

	// Step 2: Create artificial constraint points
	float d = standardDeviation;  // offset distance
	createConstraintPoints(points, normals, d);
	
	unsigned int m = constraintPoints.size();

	// Step 3: Build nearest neighbor structure
	knn.setPoints(&constraintPoints);

	// Step 4: Determine system size and adapt search radius strategy
	std::cout << "Building RBF system..." << std::endl;
	// Classification based on size - will be refined after analyzing density
	bool isLargeSystem = (m > 100000);  // 100k constraint points = ~33k original points
	float searchRadius = estimateAndAdaptSearchRadius(3.0f * c, isLargeSystem, m);
	
	// Store the adapted search radius for consistent evaluation
	radius = searchRadius;

	// Step 5: Build the sparse matrix system
	Eigen::SparseMatrix<double> A(m, m);
	Eigen::VectorXd b(m);

#if defined(_OPENMP) && defined(USE_PARALLEL_RBF)
	if (useParallel)
	{
		buildMatrixParallel(A, b, searchRadius, m);
	}
	else
#endif
	{
		buildMatrixSequential(A, b, searchRadius, m);
	}

	// Step 6: Reclassify system based on actual matrix density
	size_t nonZeros = A.nonZeros();
	double avgNonZerosPerRow = (double)nonZeros / m;
	
	// CRITICAL: Reclassify based on density, not just size
	// A matrix with 100+ neighbors per row is effectively "large" regardless of m
	if (!isLargeSystem && avgNonZerosPerRow > 100) {
		std::cout << "\nWARNING: Matrix is very dense (" << int(avgNonZerosPerRow) << " avg neighbors)" << std::endl;
		std::cout << "  Treating as LARGE SYSTEM for numerical stability." << std::endl;
		isLargeSystem = true;
	}
	
	// Step 7: Compute and apply regularization
	double lambda = computeAdaptiveRegularization(A, m, isLargeSystem);
	applyRegularization(A, lambda, m);

	// Step 8: Solve the linear system
#ifdef _OPENMP
	// Enable Eigen's parallel mode if OpenMP is available
	if (useParallel) {
		Eigen::setNbThreads(omp_get_max_threads());
	} else {
		Eigen::setNbThreads(1);
	}
#endif

	Eigen::VectorXd x;
	if (isLargeSystem) {
		x = solveLargeSystem(A, b, m);
	} else {
		x = solveSmallSystem(A, b, m);
	}

	// Step 9: Check if solver failed
	if (x.size() == 0) {
		coefficients.clear();
		return;
	}

	// Step 10: Store coefficients
	storeCoefficients(x, m);

	auto totalEndTime = std::chrono::high_resolution_clock::now();
	auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime);

	std::cout << "\n========================================" << std::endl;
	std::cout << "INITIALIZATION COMPLETE" << std::endl;
	std::cout << "========================================" << std::endl;
	std::cout << "Total time:          " << totalDuration.count() / 1000.0 << "s" << std::endl;
	std::cout << "RBF function ready for evaluation." << std::endl;
	std::cout << "========================================\n" << std::endl;
}


/* This operator returns a boolean that if true signals that the value parameter
   has been modified to contain the value of the RBF implicit distance at point P.
 */

bool RBFFunction::operator()(const glm::vec3 &P, float &value) const
{
	if (constraintPoints.empty() || coefficients.empty())
		return false;

	// Find neighbors within support radius
	vector<pair<size_t, float>> neighbors;

	// Use the SAME search radius that was used during initialization
	// This is critical for consistency!
	knn.getNeighborsInRadius(P, radius, neighbors);

	if (neighbors.empty())
	{
		// Fallback for low-resolution grids: extrapolate from nearest point
		// This prevents holes in undefined regions
		vector<size_t> nearestIdx;
		vector<float> nearestDistSq;
		unsigned int found = knn.getKNearestNeighbors(P, 1, nearestIdx, nearestDistSq);
		
		if (found > 0)
		{
			float dist = sqrt(nearestDistSq[0]);
			// Extrapolate: assume the value increases linearly beyond support radius
			// Use the value at the nearest point plus distance penalty
			value = coefficients[nearestIdx[0]] * evaluatePhi(radius) + (dist - radius);
			return true;
		}
		return false;
	}

	// Compute f(P) = sum_i phi(||P - pi||) * ci
	value = 0.0f;

	for (const auto &neighbor : neighbors)
	{
		size_t i = neighbor.first;
		float dist = sqrt(neighbor.second);  // neighbor.second is squared distance
		float phi = evaluatePhi(dist);

		if (phi != 0.0f)
		{
			value += phi * coefficients[i];
		}
	}

	return true;
}

// Gaussian RBF with compact support
float RBFFunction::evaluatePhi(float r) const
{
	// Compact support: phi(r) = exp(-r^2 / (2*c^2)) if r < 3c, else 0
	if (r < 3.0f * c)
	{
		return exp(-r * r / (2.0f * c * c));
	}
	else
	{
		return 0.0f;
	}
}





