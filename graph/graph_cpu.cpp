#include "graph_cpu.h"
#include <queue>
#include <set>

bool GraphCPU::is_connected(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return false;

	size_t graphSize = this->vertices.size();

	std::vector<bool> visited(graphSize, false);

	visited[from] = true;

	std::queue<int> q;
	q.push(from);

	while (!q.empty())
	{
		int v = q.front();
		q.pop();

		for (Edge& edge : vertices[v].edges)
		{
			if (edge.target == to)
			{
				return true;
			}

			if (!visited[edge.target])
			{
				visited[edge.target] = true;
				q.push(edge.target);
			}
		}
	}

	return false;
}
unsigned int GraphCPU::get_shortest_path(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return UINT_MAX;

	size_t graphSize = this->vertices.size();

	std::vector<unsigned int> costs(graphSize, UINT_MAX);
	costs[from] = 0;

	std::set<std::pair<double, int>> queue;
	queue.insert({ 0, from });

	while (!queue.empty())
	{
		std::pair<double, int> v = *queue.begin();
		queue.erase(queue.begin());

		if (v.second == to)
		{
			return costs[v.second];
		}

		for (Edge& edge : vertices[v.second].edges)
		{
			if (costs[v.second] + edge.cost < costs[edge.target])
			{
				queue.erase({ costs[edge.target], edge.target });
				costs[edge.target] = costs[v.second] + edge.cost;
				queue.insert({ costs[edge.target], edge.target });
			}
		}
	}

	return UINT_MAX;

}