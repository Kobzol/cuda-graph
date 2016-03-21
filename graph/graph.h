#pragma once

#include <vector>

struct Edge
{
public:
	explicit Edge(int target, unsigned int cost = 1) : target(target), cost(cost)
	{

	}

	int target;
	unsigned int cost;
};

/*
 * Graph vertex with list of it's edges.
 */
struct Vertex
{
public:
	std::vector<Edge> edges;
};

/*
 * Graph with list of it's vertices.
 */
class Graph
{
public:
	virtual int add_vertex();
	virtual void add_edge(int from, int to, unsigned int cost = 1.0);
	bool has_vertex(int id) const;

	virtual bool is_connected(int from, int to) = 0;
	virtual unsigned int get_shortest_path(int from, int to) = 0;

	std::vector<Vertex> vertices;
};
