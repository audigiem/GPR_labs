#ifndef _RBF_FUNCTION_INCLUDE
#define _RBF_FUNCTION_INCLUDE


#include <vector>
#include <Eigen/Sparse>
#include "ImplicitFunction.h"
#include "PointCloud.h"
#include "NearestNeighbors.h"


class RBFFunction : public ImplicitFunction
{

public:
	void init(const PointCloud *pointCloud, float standardDeviation, float supportRadius, bool useParallel = true);

	bool operator()(const glm::vec3 &P, float &value) const;
	
private:
	// Helper functions for init()
	void createConstraintPoints(const std::vector<glm::vec3> &points,
	                             const std::vector<glm::vec3> &normals,
	                             float offset);

	float estimateAndAdaptSearchRadius(float initialRadius, bool isLargeSystem, unsigned int m);

	void buildMatrixParallel(Eigen::SparseMatrix<double> &A,
	                         Eigen::VectorXd &b,
	                         float searchRadius,
	                         unsigned int m);

	void buildMatrixSequential(Eigen::SparseMatrix<double> &A,
	                           Eigen::VectorXd &b,
	                           float searchRadius,
	                           unsigned int m);

	double computeAdaptiveRegularization(const Eigen::SparseMatrix<double> &A,
	                                      unsigned int m,
	                                      bool isLargeSystem);

	void applyRegularization(Eigen::SparseMatrix<double> &A,
	                         double lambda,
	                         unsigned int m);

	Eigen::VectorXd solveLargeSystem(const Eigen::SparseMatrix<double> &A,
	                                  const Eigen::VectorXd &b,
	                                  unsigned int m);

	Eigen::VectorXd solveSmallSystem(const Eigen::SparseMatrix<double> &A,
	                                  const Eigen::VectorXd &b,
	                                  unsigned int m);

	void storeCoefficients(const Eigen::VectorXd &x, unsigned int m);

	float evaluatePhi(float r) const;

	const PointCloud *cloud;
	NearestNeighbors knn;
	float c;  // standard deviation
	float radius;  // support radius

	std::vector<glm::vec3> constraintPoints;  // Original + artificial points
	std::vector<float> constraintValues;  // Values at constraint points
	std::vector<float> coefficients;  // RBF coefficients
};


#endif // _RBF_FUNCTION_INCLUDE




