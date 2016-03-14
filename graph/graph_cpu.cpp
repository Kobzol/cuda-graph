#include "graph_cpu.h"
#include <queue>

bool GraphCPU::is_connected(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return false;

	size_t graphSize = this->vertices.size();

	std::vector<unsigned int> costs(graphSize, UINT_MAX);
	std::vector<bool> visited(graphSize, false);

	visited[from] = true;
	costs[from] = 0;

	std::queue<int> q;
	q.push(from);

	while (!q.empty())
	{
		int v = q.front();
		q.pop();

		for (int edge : vertices[v].edges)
		{
			if (edge == to)
			{
				return true;
			}

			if (!visited[edge])
			{
				costs[edge] = costs[v] + 1;
				visited[edge] = true;
				q.push(edge);
			}
		}
	}

	return false;
}