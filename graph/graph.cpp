#include "graph.h"

int Graph::add_vertex()
{
	this->vertices.push_back(Vertex());
	return (int) this->vertices.size() - 1;
}
void Graph::add_edge(int from, int to, unsigned int cost)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return;

	this->vertices[from].edges.push_back(Edge(to, cost));
	this->vertices[to].edges.push_back(Edge(from, cost));
}
bool Graph::has_vertex(int id) const
{
	return id >= 0 && id < this->vertices.size();
}
