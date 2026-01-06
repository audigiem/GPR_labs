#include <iostream>
#include <Eigen/Dense>
#include <glm/gtx/norm.hpp>
#include "MongePatch.h"


// Given a point P, its normal, and its closest neighbors (including itself) 
// compute a quadratic Monge patch that approximates the neighborhood of P.
// The resulting patch will be used to compute the principal curvatures of the 
// surface at point P.

void MongePatch::init(const glm::vec3 &P, const glm::vec3 &normal, const vector<glm::vec3> &closest)
{
	// Step 1. Compute local vector basis
	glm::vec3 w = - glm::normalize(normal);
	glm::vec3 u = glm::normalize(glm::cross(glm::vec3(1,0,0), w));
	if (glm::length(u) < 1e-6f)
		u = glm::normalize(glm::cross(glm::vec3(0,1,0), w));
	glm::vec3 v = glm::cross(w, u);

	// Step 2. Transform all closest points to local coordinates
	size_t m = closest.size();
	Eigen::MatrixXd Q(m, 6);
	Eigen::VectorXd W(m);
	for (long i = 0; i < m; i++) {
		glm::vec3 delta = closest[i] - P;
		double ui = glm::dot(u, delta);
		double vi = glm::dot(v, delta);
		double wi = glm::dot(w, delta);

		Q(i, 0) = ui * ui;
		Q(i, 1) = ui * vi;
		Q(i, 2) = vi * vi;
		Q(i, 3) = ui;
		Q(i, 4) = vi;
		Q(i, 5) = 1.0;
		W(i) = wi;
	}

	// Step 3. Solve least squares problem
	Eigen::MatrixXd A = Q.transpose() * Q;
	Eigen::VectorXd B = Q.transpose() * W;
	Eigen::VectorXd s = A.ldlt().solve(B);

	// Step 4. Extract Hessian coefficients
	double a = s(0);
	double b = s(1);
	double c = s(2);
	// The Hessian matrix is:
	// | 2a   b |
	// |  b  2c |
	// std::cout << "Hessian matrix:" << std::endl;
	// std::cout << "| " << 2*a << "   " << b << " |" << std::endl;
	// std::cout << "| " << b << "  " << 2*c << " |" << std::endl;
	H(0,0) = 2 * double(a);
	H(0,1) = double(b);
	H(1,0) = double(b);
	H(1,1) = 2 * double(c);
}

// Return the values of the two principal curvatures for this patch

void MongePatch::principalCurvatures(float &kmin, float &kmax) const
{
	// Step 5. Compute eigenvalues of the Hessian matrix
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
	if (solver.info() != Eigen::Success) {
		std::cerr << "Error computing eigenvalues of the Hessian matrix." << std::endl;
		kmin = 0.f;
		kmax = 0.f;
		return;
	}
	Eigen::VectorXd eigenvalues = solver.eigenvalues();
	kmin = static_cast<float>(eigenvalues(0));
	kmax = static_cast<float>(eigenvalues(1));
}


