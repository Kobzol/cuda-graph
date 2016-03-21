#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <cuda_profiler_api.h>

#include "cudamem.h"
#include "util.h"
#include "graph.h"
#include "graph_cuda.h"
#include "graph_cpu.h"

std::default_random_engine engine((unsigned int) time(nullptr));

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
			vertex.edges.push_back(Edge(edge, 1));
		}
	}
}
void load_sigmod_graph(std::string path, std::vector<Vertex>& vertices)
{
	std::fstream graphFile(path, std::ios::in);
	vertices.resize(1600000, Vertex());

	std::string line;
	while (std::getline(graphFile, line))
	{
		std::stringstream ss(line);
		int from, to;
		ss >> from >> to;

		vertices[from].edges.push_back(Edge(to, 1));
		vertices[to].edges.push_back(Edge(from, 1));
	}

	graphFile.close();
}
void load_dimacs_graph(std::string path, std::vector<Vertex>& vertices, int additionalEdges = 0)
{
	std::fstream graphFile(path, std::ios::in);
	std::uniform_int_distribution<int> randomGenerator;

	std::string line;
	while (std::getline(graphFile, line))
	{
		if (line[0] == 'c')
		{
			continue;
		}
		else if (line[0] == 'p')
		{
			std::stringstream ss(line.substr(line.find("sp") + 2));
			int verticesCount, edgeCount;
			ss >> verticesCount >> edgeCount;
			vertices.resize(verticesCount + 1, Vertex());

			randomGenerator = std::uniform_int_distribution<int>(0, (int) vertices.size() - 1);
		}
		else if (line[0] == 'a')
		{
			std::stringstream ss(line.substr(line.find("a") + 1));
			int from, to, weight;
			ss >> from >> to >> weight;
			vertices[from].edges.push_back(Edge(to, weight));

			for (int i = 0; i < additionalEdges; i++)
			{
				vertices[from].edges.push_back(Edge(randomGenerator(engine), rand() % 10000));
			}
		}
	}

	graphFile.close();
}

int main()
{
	srand((unsigned int) time(nullptr));

	GraphCUDA g;
	load_dimacs_graph("new-york.gr", g.vertices, 5);

	size_t count = 0;
	for (Vertex& vertex : g.vertices)
	{
		count += vertex.edges.size();
	}

	std::cout << "Average # of edges: " << count / (double) g.vertices.size() << std::endl;

	GraphCPU cpu;
	cpu.vertices = g.vertices;

	std::uniform_int_distribution<int> randomGenerator(0, (int) g.vertices.size() - 1);

	std::cout << "Load finished" << std::endl;

	//cudaProfilerStart();

	for (int i = 0; i < 10; i++)
	{
		int from = randomGenerator(engine);
		int to = randomGenerator(engine);

		Timer timer;
		unsigned int resultGPU = g.is_connected(from, to);
		timer.print("GPU");

		timer.start();
		unsigned int resultCPU = cpu.is_connected(from, to);
		timer.print("CPU");

		if (resultGPU != resultCPU)
		{
			std::cout << "Error at query " << i << ": expected " << resultCPU << ", got " << resultGPU << std::endl;
		}
	}

	std::cout << "Calculation finished" << std::endl;

	getchar();

	//cudaProfilerStop();

    return 0;
}
