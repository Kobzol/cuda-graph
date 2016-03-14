#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "cudamem.h"
#include "util.h"
#include "graph.h"
#include "graph_cuda.h"

void generate_graph(size_t vertexCount, size_t degree, std::vector<Vertex>& vertices)
{
	srand((unsigned int) time(nullptr));

	// generate graph
	for (size_t i = 0; i < vertexCount; i++)
	{
		vertices.push_back(Vertex());
		Vertex& vertex = vertices[vertices.size() - 1];

		size_t edgeCount = degree;

		for (size_t j = 0; j < edgeCount; j++)
		{
			int edge = rand() % vertexCount;
			if (edge == i)
			{
				j--;
				continue;
			}
			vertex.edges.push_back(edge);
		}
	}
}
void load_graph(std::vector<Vertex>& vertices)
{
	std::fstream graphFile("init-file.txt", std::ios::in);
	vertices.resize(1600000, Vertex());

	std::string line;
	while (std::getline(graphFile, line))
	{
		std::stringstream ss(line);
		int from, to;
		ss >> from >> to;

		vertices[from].edges.push_back(to);
		vertices[to].edges.push_back(from);
	}

	graphFile.close();
}

int main()
{
	cudaInit();

	GraphCUDA g;
	g.add_vertex();
	g.add_vertex();
	g.add_vertex();
	g.add_edge(0, 1);
	g.add_edge(1, 2);

	std::cout << g.is_connected(0, 2) << std::endl;


	getchar();

    return 0;
}
