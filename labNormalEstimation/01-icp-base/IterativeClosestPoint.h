#ifndef _ITERATIVE_CLOSEST_POINT_INCLUDE
#define _ITERATIVE_CLOSEST_POINT_INCLUDE


#include "PointCloud.h"
#include "NearestNeighbors.h"


class IterativeClosestPoint
{

public:
	void setClouds(PointCloud *pointCloud1, PointCloud *pointCloud2);
	
	void markBorderPoints();
	vector<int> *computeCorrespondence();
	glm::mat4 computeICPStep();
	
	vector<int> *computeFullICP(unsigned int maxSteps = 100);
	
private:
	PointCloud *cloud1, *cloud2;
	NearestNeighbors knn;
	// to store information about border points in cloud 1
	vector<bool> borderPoints;
	vector<int> correspondence;

};


#endif // _ITERATIVE_CLOSEST_POINT_INCLUDE


