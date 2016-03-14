#pragma once

#include <vector>

#include "graph.h"

class GraphCPU : public Graph
{
public:
	virtual bool is_connected(int from, int to) override;
};