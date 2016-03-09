#pragma once

#include <vector>

struct VertexCUDA
{
public:
	VertexCUDA(int edgeIndex, int edgeCount) : edgeIndex(edgeIndex), edgeCount(edgeCount), visited(false), frontier(false), frontier_next(false)
	{

	}

	int edgeIndex;
	int edgeCount;
	bool visited;
	bool frontier;
	bool frontier_next;
};

struct Vertex
{
public:
	std::vector<int> edges;
};

void linearize_vertices(const std::vector<Vertex>& vertices, std::vector<VertexCUDA>& cudaVertices, std::vector<int>& edges)
{
	for (const Vertex& vertex : vertices)
	{
		int edgeCount = (int) vertex.edges.size();
		int edgeIndex = (int) edges.size();

		for (int edge : vertex.edges)
		{
			edges.push_back(edge);
		}

		cudaVertices.emplace_back(edgeIndex, edgeCount);
	}
}