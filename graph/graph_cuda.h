#pragma once

#include "graph.h"

class GraphCUDA : public Graph
{
public:
	virtual int add_vertex() override;
	virtual void add_edge(int from, int to) override;

	virtual bool is_connected(int from, int to) override;

private:
	void relinearizeVertices();

	std::vector<int> edges;
	std::vector<LinearizedVertex> linearizedVertices;
	bool dirty = true;
};

void cudaInit();