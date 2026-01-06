#include <iostream>
#include <algorithm>
#include "IterativeClosestPoint.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <glm/gtc/matrix_transform.hpp>


void IterativeClosestPoint::setClouds(PointCloud *pointCloud1, PointCloud *pointCloud2)
{
	cloud1 = pointCloud1;
	cloud2 = pointCloud2;
	knn.setPoints(&(cloud1->getPoints()));

	// Initialize border points vector
	borderPoints.resize(cloud1->getPoints().size(), false);
}

// Mark border points using PCA and angle analysis as described in course notes
void IterativeClosestPoint::markBorderPoints()
{
	const vector<glm::vec3>& points = cloud1->getPoints();
	const int K = 15; // Number of neighbors
	const float beta = 3.0f * M_PI / 4.0f; // Angle threshold (135 degrees)

	vector<size_t> neighbors;
	vector<float> dists_squared;

	for (size_t i = 0; i < points.size(); ++i) {
		// Step 1: Find K nearest neighbors
		knn.getKNearestNeighbors(points[i], K, neighbors, dists_squared);

		// Step 2: PCA to get local frame (same approach as NormalEstimator)
		// Compute centroid of neighbors
		glm::vec3 centroid(0.0f);
		for (const auto& idx : neighbors) {
			centroid += points[idx];
		}
		centroid /= static_cast<float>(neighbors.size());

		// Compute covariance matrix (same as NormalEstimator)
		Eigen::Matrix3f covMatrix;
		covMatrix.setZero();

		for (const auto& idx : neighbors) {
			glm::vec3 p = points[idx] - centroid;
			Eigen::Vector3f point;
			point << p.x, p.y, p.z;
			covMatrix += point * point.transpose();
		}

		// Compute eigenvectors (same as NormalEstimator)
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covMatrix);
		if (solver.info() != Eigen::Success) continue;

		// Get eigenvectors: v1 (largest), v2 (middle), v3 (smallest - normal)
		Eigen::Vector3f v1 = solver.eigenvectors().col(2); // Largest eigenvalue
		Eigen::Vector3f v2 = solver.eigenvectors().col(1); // Middle eigenvalue

		// Step 3: Project neighbors to local 2D frame and compute angles
		vector<float> angles;
		for (const auto& idx : neighbors) {
			glm::vec3 diff = points[idx] - points[i];

			// Project to 2D using v1 and v2 (equation 4.1 from course)
			float x_prime = diff.x * v1(0) + diff.y * v1(1) + diff.z * v1(2);
			float y_prime = diff.x * v2(0) + diff.y * v2(1) + diff.z * v2(2);

			// Step 4: Compute angle (equation 4.2 from course)
			float angle = atan2(y_prime, x_prime);
			angles.push_back(angle);
		}

		// Step 5: Sort angles and compute differences
		sort(angles.begin(), angles.end());

		float maxAngleDiff = 0.0f;
		for (size_t j = 0; j < angles.size(); ++j) {
			float diff;
			if (j == angles.size() - 1) {
				// Wrap around: difference between last and first
				diff = (2.0f * M_PI) + angles[0] - angles[j];
			} else {
				diff = angles[j + 1] - angles[j];
			}
			maxAngleDiff = max(maxAngleDiff, diff);
		}

		// Step 6: Check if border point
		if (maxAngleDiff >= beta) {
			borderPoints[i] = true;
			// Change color to red for visualization
			cloud1->getColors()[i] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		}
	}
}

// Compute closest point in cloud1 for each non-border point in cloud2
vector<int> *IterativeClosestPoint::computeCorrespondence()
{
	const vector<glm::vec3>& points1 = cloud1->getPoints();
	const vector<glm::vec3>& points2 = cloud2->getPoints();

	correspondence.clear();
	correspondence.resize(points2.size(), -1);

	vector<size_t> neighbors;
	vector<float> dists_squared;

	for (size_t i = 0; i < points2.size(); ++i) {
		// Find nearest neighbor in cloud1
		knn.getKNearestNeighbors(points2[i], 1, neighbors, dists_squared);

		// Check if the nearest neighbor is not a border point
		if (neighbors.size() > 0 && !borderPoints[neighbors[0]]) {
			correspondence[i] = neighbors[0];
		}
	}

	return &correspondence;
}

// Compute rotation and translation matrix using SVD
glm::mat4 IterativeClosestPoint::computeICPStep()
{
	const vector<glm::vec3>& points1 = cloud1->getPoints();
	const vector<glm::vec3>& points2 = cloud2->getPoints();

	// Collect valid correspondences
	vector<glm::vec3> srcPoints, dstPoints;
	for (size_t i = 0; i < correspondence.size(); ++i) {
		if (correspondence[i] >= 0) {
			srcPoints.push_back(points2[i]);
			dstPoints.push_back(points1[correspondence[i]]);
		}
	}

	if (srcPoints.size() < 3) {
		return glm::mat4(1.0f); // Not enough correspondences
	}

	// Compute centroids
	glm::vec3 centroidSrc(0.0f), centroidDst(0.0f);
	for (size_t i = 0; i < srcPoints.size(); ++i) {
		centroidSrc += srcPoints[i];
		centroidDst += dstPoints[i];
	}
	centroidSrc /= static_cast<float>(srcPoints.size());
	centroidDst /= static_cast<float>(dstPoints.size());

	// Center the point sets
	Eigen::MatrixXf P(3, srcPoints.size());
	Eigen::MatrixXf Q(3, srcPoints.size());

	for (size_t i = 0; i < srcPoints.size(); ++i) {
		glm::vec3 p = srcPoints[i] - centroidSrc;
		glm::vec3 q = dstPoints[i] - centroidDst;

		P(0, i) = p.x; P(1, i) = p.y; P(2, i) = p.z;
		Q(0, i) = q.x; Q(1, i) = q.y; Q(2, i) = q.z;
	}

	// Compute covariance matrix H = P * Q^T
	Eigen::Matrix3f H = P * Q.transpose();

	// Compute SVD: H = U * S * V^T
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f V = svd.matrixV();

	// Compute rotation matrix R = V * U^T
	Eigen::Matrix3f R = V * U.transpose();

	// Handle reflection case
	if (R.determinant() < 0) {
		V.col(2) *= -1;
		R = V * U.transpose();
	}

	// Compute translation t = centroidDst - R * centroidSrc
	Eigen::Vector3f src(centroidSrc.x, centroidSrc.y, centroidSrc.z);
	Eigen::Vector3f dst(centroidDst.x, centroidDst.y, centroidDst.z);
	Eigen::Vector3f t = dst - R * src;

	// Build 4x4 transformation matrix
	glm::mat4 transform(1.0f);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			transform[j][i] = R(i, j);
		}
		transform[3][i] = t(i);
	}

	// Apply transformation to cloud2
	cloud2->transform(transform);

	return transform;
}

// Perform full ICP algorithm with convergence criteria
vector<int> *IterativeClosestPoint::computeFullICP(unsigned int maxSteps)
{
	const float convergenceThreshold = 1e-5f;
	vector<int> prevCorrespondence;

	for (unsigned int step = 0; step < maxSteps; ++step) {
		// Compute correspondence
		computeCorrespondence();

		// Check if correspondence has changed
		if (step > 0 && correspondence == prevCorrespondence) {
			std::cout << "ICP converged: correspondence unchanged (step " << step << ")" << std::endl;
			break;
		}
		prevCorrespondence = correspondence;

		// Compute and apply ICP step
		glm::mat4 transform = computeICPStep();

		// Compute Frobenius norm of (transform - identity)
		glm::mat4 identity(1.0f);
		glm::mat4 diff = transform - identity;
		float frobeniusNorm = 0.0f;
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				frobeniusNorm += diff[i][j] * diff[i][j];
			}
		}
		frobeniusNorm = sqrt(frobeniusNorm);

		// Check convergence
		if (frobeniusNorm < convergenceThreshold) {
			std::cout << "ICP converged: transformation norm below threshold (step " << step << ")" << std::endl;
			break;
		}

		if (step == maxSteps - 1){
		    std::cout << "ICP did not converge in " <<  maxSteps << " steps" << std::endl;
		}
	}



	return &correspondence;
}