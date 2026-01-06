#ifndef _LAPLACIAN_SMOOTHING_INCLUDE
#define _LAPLACIAN_SMOOTHING_INCLUDE


#include "TriangleMesh.h"


class LaplacianSmoothing
{
public:
	void setMesh(TriangleMesh *newMesh);
	void iterativeLaplacian(int nIterations, float lambda);
	void iterativeBilaplacian(int nIterations, float lambda);
	void iterativeLambdaNu(int nIterations, float lambda);
	
	void globalLaplacian(const vector<bool> &constraints, bool showConstraints = true);
	void globalBilaplacian(const vector<bool> &constraints, float constraintWeight, bool showConstraints = true);
	
private:
	TriangleMesh *mesh;
	
};


#endif // _LAPLACIAN_SMOOTHING_INCLUDE


