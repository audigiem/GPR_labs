#include "NormalEstimator.h"
#include "NearestNeighbors.h"
#include <Eigen/Eigenvalues>

// This method has to compute a normal per point in the 'points' vector and put it in the 
// 'normals' vector. The 'normals' vector already has the same size as the 'points' vector. 
// There is no need to push_back new elements, only to overwrite them ('normals[i] = ...;')
// In order to compute the normals you should use PCA. The provided 'NearestNeighbors' class
// wraps the nanoflann library that computes K-nearest neighbors efficiently.

void NormalEstimator::computePointCloudNormals(const vector<glm::vec3> &points, vector<glm::vec3> &normals)
{
    // Neighbors search structure
    NearestNeighbors nn;
    nn.setPoints(&points);
    const int K = 10; // Number of neighbors to use for normal estimation
    vector<size_t> neighbors;
    vector<float> dists_squared;

    // For each point, compute the normal using PCA on its K nearest neighbors
    for (size_t i = 0; i < points.size(); ++i) {
        // Find K nearest neighbors
        nn.getKNearestNeighbors(points[i], K, neighbors, dists_squared);

        // 1. Compute the centroid of the neighbors
        glm::vec3 centroid(0.0f);
        for (const auto& idx : neighbors) {
            centroid += points[idx];
        }
        centroid /= static_cast<float>(neighbors.size());

        // 2. Compute the covariance matrix
        Eigen::Matrix3f covMatrix;
        covMatrix.setZero();

        for (const auto& idx : neighbors) {
            glm::vec3 p = points[idx] - centroid;

            Eigen::Vector3f point;
            point << p.x, p.y, p.z;

            covMatrix += point * point.transpose();
        }

        // 3. Compute eigenvalues and eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covMatrix);

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Error computing eigenvalues");
        }

        // The normal is the eigenvector corresponding to the smallest eigenvalue
        Eigen::Vector3f eigenVector = solver.eigenvectors().col(0);

        //Convert en glm::vec3
        normals[i] = glm::normalize(glm::vec3(eigenVector(0), eigenVector(1), eigenVector(2)));

        //
        if (glm::dot(normals[i], points[i] - centroid) < 0.0f) {
            normals[i] = -normals[i];
        }
    }

}


