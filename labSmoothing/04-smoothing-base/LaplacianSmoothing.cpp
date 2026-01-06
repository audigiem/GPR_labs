// LaplacianSmoothing.cpp
#include "LaplacianSmoothing.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <iostream>
#include <string>
#include <vector>

// Ensure TriangleMesh, glm::vec3 are available via your headers included in the
// header.

void LaplacianSmoothing::setMesh(TriangleMesh *newMesh) { mesh = newMesh; }

/* Iterative uniform Laplacian:
   p' = p + lambda * delta(p)
   where delta(pi) = (1/|Ni|) sum_{j in Ni} (pj - pi)
*/
void LaplacianSmoothing::iterativeLaplacian(int nIterations, float lambda) {
    std::cout << "Start iterative Laplacian with " << nIterations << " iterations and lambda=" <<  lambda << std::endl; 
    auto &vertices = mesh->getVertices();
    const size_t n = vertices.size();
    if (n == 0)
    return;

    std::vector<glm::vec3> newVertices(n);

    for (int iter = 0; iter < nIterations; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            std::vector<unsigned int> neighbors;
            mesh->getNeighbors((unsigned int)i, neighbors);

            glm::vec3 laplacian(0.0f);
            if (!neighbors.empty()) {
                for (unsigned int j : neighbors) {
                    laplacian += vertices[j] - vertices[i];
                }
                laplacian /= static_cast<float>(neighbors.size());
            }
            newVertices[i] = vertices[i] + lambda * laplacian;
        }
        vertices = newVertices;
    }
}

/* Iterative bilaplacian: two-step process to avoid volume loss
   Step 1: p' = p + lambda * delta(p)
   Step 2: p'' = p' - lambda * delta(p')  (note the minus sign)
*/
void LaplacianSmoothing::iterativeBilaplacian(int nIterations, float lambda) {
    std::cout << "Start iterative BiLaplacian (two-step) with " << nIterations << " iterations and lambda=" <<  lambda << std::endl;
    auto &vertices = mesh->getVertices();
    const size_t n = vertices.size();
    if (n == 0)
        return;

    std::vector<glm::vec3> tempVertices(n);
    std::vector<glm::vec3> newVertices(n);

    for (int iter = 0; iter < nIterations; ++iter) {
        // Step 1: p' = p + lambda * delta(p)
        for (size_t i = 0; i < n; ++i) {
            std::vector<unsigned int> neighbors;
            mesh->getNeighbors((unsigned int)i, neighbors);

            glm::vec3 lap(0.0f);
            if (!neighbors.empty()) {
                for (unsigned int j : neighbors) {
                    lap += vertices[j] - vertices[i];
                }
                lap /= static_cast<float>(neighbors.size());
            }
            tempVertices[i] = vertices[i] + lambda * lap;
        }

        // Step 2: p'' = p' - lambda * delta(p')
        for (size_t i = 0; i < n; ++i) {
            std::vector<unsigned int> neighbors;
            mesh->getNeighbors((unsigned int)i, neighbors);

            glm::vec3 lap(0.0f);
            if (!neighbors.empty()) {
                for (unsigned int j : neighbors) {
                    lap += tempVertices[j] - tempVertices[i];
                }
                lap /= static_cast<float>(neighbors.size());
            }
            newVertices[i] = tempVertices[i] - lambda * lap;
        }

        vertices = newVertices;
    }
}

/* Taubin's operator: apply lambda step then mu step.
   Use the relation from the lab: 1/lambda + 1/mu ≈ 0.1  => mu = lambda /
   (0.1*lambda - 1) (for typical lambda > 0 this gives mu < 0).
*/
void LaplacianSmoothing::iterativeLambdaNu(int nIterations, float lambda) {
  std::cout << "Start iterative Lambda-Nu with " << nIterations << " iterations and lambda=" <<  lambda << std::endl;
  auto &vertices = mesh->getVertices();
  const size_t n = vertices.size();
  if (n == 0)
    return;

  std::vector<glm::vec3> newVertices(n);

  // compute mu according to the formula in the lab using double precision for stability
  double lam = static_cast<double>(lambda);
  double denom = 0.1 * lam - 1.0;
  double mu = 0.0;
  const double eps = 1e-8;

  if (std::abs(denom) < eps) {
    // fallback to conservative negative value
    mu = -lam * 0.5;
  } else {
    mu = lam / denom;
  }

  // clamp mu to avoid extreme values
  const double maxMag = 1e2;
  if (mu > maxMag) mu = maxMag;
  if (mu < -maxMag) mu = -maxMag;

  for (int iter = 0; iter < nIterations; ++iter) {
    // lambda step: p' = p + lambda * delta(p)
    for (size_t i = 0; i < n; ++i) {
      std::vector<unsigned int> neighbors;
      mesh->getNeighbors((unsigned int)i, neighbors);

      glm::vec3 lap(0.0f);
      if (!neighbors.empty()) {
        for (unsigned int j : neighbors) {
          lap += vertices[j] - vertices[i];
        }
        lap /= static_cast<float>(neighbors.size());
      }
      newVertices[i] = vertices[i] + lambda * lap;
    }
    vertices = newVertices;

    // mu step: p'' = p' + mu * delta(p')
    for (size_t i = 0; i < n; ++i) {
      std::vector<unsigned int> neighbors;
      mesh->getNeighbors((unsigned int)i, neighbors);

      glm::vec3 lap(0.0f);
      if (!neighbors.empty()) {
        for (unsigned int j : neighbors) {
          lap += vertices[j] - vertices[i];
        }
        lap /= static_cast<float>(neighbors.size());
      }
      newVertices[i] = vertices[i] + static_cast<float>(mu) * lap;
    }
    vertices = newVertices;
  }
}

/* Global Laplacian smoothing:
   Build A x = b where unconstrained rows encode delta(p') = 0 and constrained
   rows encode p' = original. For an unconstrained vertex i: -p_i' + (1/|Ni|)
   sum_{j in Ni} p_j' = 0
   => row i: A(i,i) = -1, A(i,j) = 1/|Ni| for j in Ni, b_i = 0
   For constrained i: A(i,i)=1, b_i = original position
*/
void LaplacianSmoothing::globalLaplacian(const std::vector<bool> &constraints, bool showConstraints) {
  std::cout << "Start global Laplacian smoothing" << std::endl;
  auto &vertices = mesh->getVertices();
  const size_t n = vertices.size();
  if (n == 0)
    return;
  if (constraints.size() != n) {
    std::cerr << "globalLaplacian: constraints size mismatch\n";
    return;
  }

  // Visualize constraints: set colors for constrained vs unconstrained vertices
  auto &colors = mesh->getColors();
  if (colors.size() != n) {
    colors.resize(n);
  }
  
  if (showConstraints) {
    for (size_t i = 0; i < n; ++i) {
      if (constraints[i]) {
        colors[i] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red for constrained
      } else {
        colors[i] = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);  // Light gray for unconstrained
      }
    }
  } else {
    // Reset to default color when not showing constraints
    for (size_t i = 0; i < n; ++i) {
      colors[i] = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);  // Default gray
    }
  }

  typedef Eigen::Triplet<double> T;
  std::vector<T> triplets;
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero((int)n, 3);

  // Build Laplacian-like rows. Treat isolated vertices as constrained to avoid singularities.
  for (size_t i = 0; i < n; ++i) {
    if (constraints[i]) {
      triplets.emplace_back((int)i, (int)i, 1.0);
      B.row((int)i) =
          Eigen::Vector3d(vertices[i].x, vertices[i].y, vertices[i].z);
    } else {
      std::vector<unsigned int> neighbors;
      mesh->getNeighbors((unsigned int)i, neighbors);

      if (neighbors.empty()) {
        // Isolated vertex: treat as constrained to avoid division by zero
        triplets.emplace_back((int)i, (int)i, 1.0);
        B.row((int)i) =
            Eigen::Vector3d(vertices[i].x, vertices[i].y, vertices[i].z);
      } else {
        const double invk = 1.0 / static_cast<double>(neighbors.size());

        // diagonal -1
        triplets.emplace_back((int)i, (int)i, -1.0);

        // off-diagonals 1/|Ni|
        for (unsigned int j : neighbors) {
          triplets.emplace_back((int)i, (int)j, invk);
        }
        // b row remains zero
      }
    }
  }

  Eigen::SparseMatrix<double> A((int)n, (int)n);
  A.setFromTriplets(triplets.begin(), triplets.end());

  // Solve for each dimension using double precision
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  if (solver.info() != Eigen::Success) {
    std::cerr << "globalLaplacian: decomposition failed\n";
    return;
  }

  for (int dim = 0; dim < 3; ++dim) {
    Eigen::VectorXd b = B.col(dim);
    Eigen::VectorXd x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
      std::cerr << "globalLaplacian: solving failed\n";
      return;
    }
    for (size_t i = 0; i < n; ++i) {
      vertices[i][dim] = static_cast<float>(x[(int)i]);
    }
  }
  std::cout << "globalLaplacian smoothing: done" << std::endl;
}

/* Global bilaplacian via least-squares:
   We build the uniform Laplacian matrix L (n x n) where for each row i:
     L(i,i) = -1
     L(i,j) = 1/|Ni|  for j in Ni
   Then minimize ||L P'||^2 + constraintWeight * ||P'_c - P_c||^2 for
   constrained vertices. This leads to the normal equations: (L^T L + W) P' = W
   * P_original where W is diagonal with constraintWeight on constrained indices
   (0 otherwise).
*/
void LaplacianSmoothing::globalBilaplacian(const std::vector<bool> &constraints,
                                           float constraintWeight, bool showConstraints) {
  std::cout<< "Start global BiLaplacian smoothing with constraint weight " << constraintWeight << std::endl;
  auto &vertices = mesh->getVertices();
  const size_t n = vertices.size();
  if (n == 0)
    return;
  if (constraints.size() != n) {
    std::cerr << "globalBilaplacian: constraints size mismatch\n";
    return;
  }

  // Visualize constraints: set colors for constrained vs unconstrained vertices
  auto &colors = mesh->getColors();
  if (colors.size() != n) {
    colors.resize(n);
  }
  
  if (showConstraints) {
    for (size_t i = 0; i < n; ++i) {
      if (constraints[i]) {
        colors[i] = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red for constrained
      } else {
        colors[i] = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);  // Light gray for unconstrained
      }
    }
  } else {
    // Reset to default color when not showing constraints
    for (size_t i = 0; i < n; ++i) {
      colors[i] = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);  // Default gray
    }
  }

  typedef Eigen::Triplet<double> T;
  std::vector<T> lapTriplets;

  // Build L (n x n). For isolated vertices, make that row zero.
  for (size_t i = 0; i < n; ++i) {
    std::vector<unsigned int> neighbors;
    mesh->getNeighbors((unsigned int)i, neighbors);

    if (neighbors.empty()) {
      // Isolated vertex: zero row (no laplacian contribution)
      // Will be stabilized by constraint weight below
    } else {
      const double invk = 1.0 / static_cast<double>(neighbors.size());

      lapTriplets.emplace_back((int)i, (int)i, -1.0);
      for (unsigned int j : neighbors) {
        lapTriplets.emplace_back((int)i, (int)j, invk);
      }
    }
  }

  Eigen::SparseMatrix<double> L((int)n, (int)n);
  L.setFromTriplets(lapTriplets.begin(), lapTriplets.end());

  // Build A = L^T L
  Eigen::SparseMatrix<double> A = L.transpose() * L;

  // Add diagonal constraint weights: A(i,i) += constraintWeight for constrained vertices
  std::vector<T> addTriplets;
  for (size_t i = 0; i < n; ++i) {
    if (constraints[i]) {
      addTriplets.emplace_back((int)i, (int)i, static_cast<double>(constraintWeight));
    } else {
      // If isolated and unconstrained, add tiny diagonal to avoid singularity
      std::vector<unsigned int> neighbors;
      mesh->getNeighbors((unsigned int)i, neighbors);
      if (neighbors.empty()) {
        addTriplets.emplace_back((int)i, (int)i, 1e-8);
      }
    }
  }

  if (!addTriplets.empty()) {
    Eigen::SparseMatrix<double> W((int)n, (int)n);
    W.setFromTriplets(addTriplets.begin(), addTriplets.end());
    A += W;
  }

  // Build RHS B = W * P_original (only non-zero for constrained rows)
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero((int)n, 3);
  if (constraintWeight > 0.0f) {
    for (size_t i = 0; i < n; ++i) {
      if (constraints[i]) {
        B.row((int)i) =
            constraintWeight *
            Eigen::Vector3d(vertices[i].x, vertices[i].y, vertices[i].z);
      }
    }
  }

  // Solve A x = B for each dimension using double precision
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.analyzePattern(A);
  solver.factorize(A);
  if (solver.info() != Eigen::Success) {
    std::cerr << "globalBilaplacian: decomposition failed\n";
    return;
  }

  for (int dim = 0; dim < 3; ++dim) {
    Eigen::VectorXd b = B.col(dim);
    Eigen::VectorXd x = solver.solve(b);
    if (solver.info() != Eigen::Success) {
      std::cerr << "globalBilaplacian: solving failed\n";
      return;
    }
    for (size_t i = 0; i < n; ++i) {
      vertices[i][dim] = static_cast<float>(x[(int)i]);
    }
  }
    std::cout << "globalBiLaplacian smoothing: done" << std::endl;
}
