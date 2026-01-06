#include "SimpleDistance.h"


/* Initialize everything to be able to compute the implicit distance of [Hoppe92] 
   at arbitrary points that are close enough to the point cloud.
 */

void SimpleDistance::init(const PointCloud *pointCloud, float samplingRadius)
{
	cloud = pointCloud;
	knn.setPoints(&pointCloud->getPoints());
	radius = samplingRadius;
}


/* This operator returns a boolean that if true signals that the value parameter
   has been modified to contain the value of the implicit function of [Hoppe92]
   at point P.
 */

bool SimpleDistance::operator()(const glm::vec3 &P, float &value) const
{
	// Find the closest point pi to P
	vector<size_t> neighbors;
	vector<float> dists_squared;

	if (knn.getKNearestNeighbors(P, 1, neighbors, dists_squared) == 0)
		return false;

	size_t i = neighbors[0];
	const glm::vec3& pi = cloud->getPoints()[i];
	const glm::vec3& ni = cloud->getNormals()[i];

	// Compute z as the projection of P onto the tangent plane at pi
	// z = pi - ((P - pi) · ni) · ni
	glm::vec3 diff = P - pi;
	float dotProduct = glm::dot(diff, ni);
	glm::vec3 z = pi + diff - dotProduct * ni;

	// Check if distance(z, pi) <= radius (ρ + δ)
	float distZ = glm::distance(z, pi);

	if (distZ <= radius)
	{
		// f(P) = (P - pi) · ni
		value = dotProduct;
		return true;
	}
	else
	{
		// f(P) = undefined
		return false;
	}
}






