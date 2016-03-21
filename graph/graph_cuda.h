#pragma once

#include "graph.h"

#define CUDA_NOT_VISITED (-1)
#define CUDA_VISITED (INT_MAX)

/*
* Linearized vertex without edges that is used in CUDA kernels.
*/
struct LinearizedVertex
{
public:
	LinearizedVertex(int edgeIndex, int edgeCount) : edgeCount(edgeCount), edgeIndex(edgeIndex), visitIndex(CUDA_NOT_VISITED)
	{

	}

	int edgeCount;
	int edgeIndex;
	int visitIndex;
};

class GraphCUDA : public Graph
{
public:
	virtual int add_vertex() override;
	virtual void add_edge(int from, int to, unsigned int cost = 1.0) override;

	virtual bool is_connected(int from, int to) override;
	virtual unsigned int get_shortest_path(int from, int to) override;

private:
	void relinearizeVertices(bool force = false);
	void initCuda();

	static bool CudaInitialized;

	std::vector<Edge> edges;
	std::vector<LinearizedVertex> linearizedVertices;
	bool dirty = true;
};

void cudaInit();