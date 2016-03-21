#pragma once

#include <vector>

#include "graph.h"

class GraphCPU : public Graph
{
public:
	virtual bool is_connected(int from, int to) override;
	virtual unsigned int get_shortest_path(int from, int to) override;
};