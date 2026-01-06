#include <iostream>
#include <cstdlib>
#include <set>
#include <map>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "Parameterizer.h"
#include "timing.h"


using namespace std;


// This method should compute new texture coordinates for the input mesh
// using the harmonic coordinates approach with uniform weights.
// The 'TriangleMesh' class has a method 'getTexCoords' that may be used 
// to access and update its texture coordinates per vertex.

void Parameterizer::harmonicCoordinates(TriangleMesh *mesh)
{
	// First we need to determine the border edges of the input mesh
	set<pair<int, int> > halfEdges;
	const vector<unsigned int> &triangles = mesh->getTriangles();
	int numTriangles = triangles.size() / 3;

	for (int t = 0; t < numTriangles; t++)
	{
		// Look for half edges H in the triangle
		for (int i = 0; i < 3; i++)
		{
			int v0 = triangles[t * 3 + i];
			int v1 = triangles[t * 3 + (i + 1) % 3];
			pair<int, int> he(v0, v1);
			pair<int, int> heOpp(v1, v0);

			// If the opposite half edge is already in the set, remove it
			if (halfEdges.find(heOpp) != halfEdges.end())
				halfEdges.erase(heOpp);
			else
				halfEdges.insert(he);
		}
	}
	
	// Then, these edges need to be arranged into a single cycle, the border polyline
	if (halfEdges.empty())
	{
		cerr << "Error: The input mesh has no border edges." << endl;
		return;
	}

	cout << "Found " << halfEdges.size() << " border half-edges." << endl;
	std::map<int, int> borderPolyline;
	for (const auto &he : halfEdges)
	{
		borderPolyline[he.first] = he.second;
	}
	// Chose first vertex of the border polyline as starting point
	int startVrtx = borderPolyline.begin()->first;
	vector<int> borderVertices;
	borderVertices.push_back(startVrtx);
	int currentVrtx = startVrtx;
	bool closed = false;
	int safety = 0;
	int maxSteps = static_cast<int>(halfEdges.size()) + 10;
	while (!closed && safety < maxSteps) {
		auto it =  borderPolyline.find(currentVrtx);
		if (it == borderPolyline.end()) {
			// Incomplete border, bad configuration
			break;
		}
		int nxt = it->second;
		if (nxt == startVrtx) {
			closed = true;
			// Closed cycle
			break;
		}
		// protect against infinite loop
		safety++;
		borderVertices.push_back(nxt);
		currentVrtx = nxt;
	}
	// If cycle is closed, explicitely add first vertex
	if (closed) {
		borderVertices.push_back(startVrtx);
		cout << "Border cycle closed with " << borderVertices.size() << " vertices." << endl;
	}
	else {
		cerr << "Error: Border cycle is not closed! Found " << borderVertices.size() << " vertices." << endl;
		return;
	}

	// Each of the vertices on the border polyline will receive a texture coordinate
	// on the border of the parameter space ([0,0] -> [1,0] -> [1,1] -> [0,1]), using 
	// the chord length approach

	// Compute total length of the border edge cycle
	float totalLength = 0.0f;
	const vector<glm::vec3> &vertices = mesh->getVertices();
	vector<float> cumulativeLength(borderVertices.size(), 0.0f);

	for (size_t i = 0; i < borderVertices.size() - 1; i++)
	{
		glm::vec3 v0 = vertices[borderVertices[i]];
		glm::vec3 v1 = vertices[borderVertices[i + 1]];
		float edgeLength = glm::length(v1 - v0);
		totalLength += edgeLength;
		cumulativeLength[i + 1] = totalLength;
	}

	// Assign 2D parameter coordinates to border vertices
	vector<glm::vec2> &texCoords = mesh->getTexCoords();
	texCoords.clear();
	texCoords.resize(vertices.size(), glm::vec2(0.0f, 0.0f));

	// Mark which vertices are on the border
	set<int> borderVertexSet;
	for (size_t i = 0; i < borderVertices.size() - 1; i++)  // -1 to avoid duplicate (start = end)
	{
		borderVertexSet.insert(borderVertices[i]);
	}

	// Assign border vertex coordinates
	for (size_t i = 0; i < borderVertices.size() - 1; i++)
	{
		int vIdx = borderVertices[i];
		float lambda = cumulativeLength[i] / totalLength;

		glm::vec2 uv;
		if (lambda < 0.25f)
			uv = glm::vec2(4.0f * lambda, 0.0f);
		else if (lambda < 0.5f)
			uv = glm::vec2(1.0f, 4.0f * (lambda - 0.25f));
		else if (lambda < 0.75f)
			uv = glm::vec2(1.0f - 4.0f * (lambda - 0.5f), 1.0f);
		else
			uv = glm::vec2(0.0f, 1.0f - 4.0f * (lambda - 0.75f));

		texCoords[vIdx] = uv;
	}

	// We build an equation system to compute the harmonic coordinates of the 
	// interior vertices

	// Create mapping from vertex index to row in the system
	int numVertices = vertices.size();
	int numInteriorVertices = numVertices - borderVertexSet.size();

	map<int, int> vertexToRow;
	int row = 0;
	for (int i = 0; i < numVertices; i++)
	{
		if (borderVertexSet.find(i) == borderVertexSet.end())
		{
			vertexToRow[i] = row++;
		}
	}

	// Build the sparse Laplacian matrix for interior vertices
	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	Eigen::VectorXd bu(numInteriorVertices);
	Eigen::VectorXd bv(numInteriorVertices);
	bu.setZero();
	bv.setZero();

	for (int vIdx = 0; vIdx < numVertices; vIdx++)
	{
		if (borderVertexSet.find(vIdx) != borderVertexSet.end())
			continue;  // Skip border vertices

		int i = vertexToRow[vIdx];

		// Get neighbors of this vertex
		vector<unsigned int> neighbors;
		mesh->getNeighbors(vIdx, neighbors);

		// Uniform weights: w_ij = 1 for all neighbors
		int degree = neighbors.size();
		tripletList.push_back(T(i, i, degree));

		for (unsigned int nIdx : neighbors)
		{
			if (borderVertexSet.find(nIdx) != borderVertexSet.end())
			{
				// Neighbor is on border, move to right-hand side
				bu(i) += texCoords[nIdx].x;
				bv(i) += texCoords[nIdx].y;
			}
			else
			{
				// Neighbor is interior, add to matrix
				int j = vertexToRow[nIdx];
				tripletList.push_back(T(i, j, -1.0));
			}
		}
	}

	// Create sparse matrix and solve
	Eigen::SparseMatrix<double> L(numInteriorVertices, numInteriorVertices);
	L.setFromTriplets(tripletList.begin(), tripletList.end());

	// Use iterative solver (BiCGSTAB)
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
	solver.compute(L);

	if (solver.info() != Eigen::Success)
	{
		cerr << "Error: Failed to factorize the Laplacian matrix." << endl;
		return;
	}

	Eigen::VectorXd u = solver.solve(bu);
	Eigen::VectorXd v = solver.solve(bv);

	if (solver.info() != Eigen::Success)
	{
		cerr << "Error: Failed to solve the linear system." << endl;
		return;
	}

	// Finally, we solve the system and assign the computed texture coordinates
	// to their corresponding vertices on the input mesh
	for (int vIdx = 0; vIdx < numVertices; vIdx++)
	{
		if (borderVertexSet.find(vIdx) == borderVertexSet.end())
		{
			int i = vertexToRow[vIdx];
			texCoords[vIdx] = glm::vec2(u(i), v(i));
		}
	}

	// Compute statistics
	glm::vec2 minUV(1e10f), maxUV(-1e10f);
	for (const auto &tc : texCoords)
	{
		minUV = glm::min(minUV, tc);
		maxUV = glm::max(maxUV, tc);
	}

	cout << "Harmonic parameterization completed: " << numInteriorVertices
	     << " interior vertices, " << borderVertexSet.size() << " border vertices." << endl;
	cout << "TexCoord range: u=[" << minUV.x << ", " << maxUV.x << "], v=[" << minUV.y << ", " << maxUV.y << "]" << endl;
}






