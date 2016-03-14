#pragma once

#include <vector>

struct LinearizedVertex
{
public:
	LinearizedVertex(int edgeIndex, int edgeCount) : edgeIndex(edgeIndex), edgeCount(edgeCount), visited(false), frontier(false), frontier_next(false)
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

class Graph
{
public:
	virtual int add_vertex();
	virtual void add_edge(int from, int to);
	bool has_vertex(int id) const;

	virtual bool is_connected(int from, int to) = 0;
	virtual double get_shortest_path(int from, int to) { return 0.0; }	// TODO

protected:
	std::vector<Vertex> vertices;
};
