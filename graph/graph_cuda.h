#pragma once

#include "graph.h"

class GraphCUDA : public Graph
{
public:
	virtual int add_vertex() override;
	virtual void add_edge(int from, int to, unsigned int cost = 1.0) override;

	virtual bool is_connected(int from, int to) override;
	virtual unsigned int get_shortest_path(int from, int to) override;

private:
	void relinearizeVertices();
	void initCuda();

	static bool CudaInitialized;

	std::vector<Edge> edges;
	std::vector<LinearizedVertex> linearizedVertices;
	bool dirty = true;
};

void cudaInit();