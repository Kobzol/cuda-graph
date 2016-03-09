#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <queue>
#include <unordered_set>
#include <vector>

#include "cuda_helper.h"
#include "graph.h"


__global__ void bfsKernel(VertexCUDA* vertices, int* edges, unsigned int* costs, size_t size)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.x + threadIdx.y;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].frontier)
	{
		vertices[pos].frontier = false;
		int distance = costs[pos];

		for (size_t i = 0; i < vertices[pos].edgeCount; i++)
		{
			int edge = edges[vertices[pos].edgeIndex + i];
			if (!vertices[edge].visited)
			{
				costs[edge] = distance + 1;
				vertices[edge].frontier_next = true;
				vertices[edge].visited = true;
			}
		}
	}
}
__global__ void bfsKernelRequeue(VertexCUDA* vertices, size_t size, bool *stop)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.x + threadIdx.y;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].frontier_next)
	{
		vertices[pos].frontier = true;
		vertices[pos].frontier_next = false;
		*stop = false;
	}
}

void bfsCuda(const std::vector<Vertex>& vertices, size_t source, std::vector<unsigned int>& costs)
{
	if (vertices.size() < 1) return;

	int graphSize = (int) vertices.size();

	std::vector<int> edges;
	std::vector<VertexCUDA> linearizedVertices;

	Timer timer;
	linearize_vertices(vertices, linearizedVertices, edges);
	timer.print();

	if (edges.size() < 1) return;

	linearizedVertices[source].frontier = true;
	linearizedVertices[source].visited = true;

	timer.start();
	CudaMemory<VertexCUDA> verticesCuda(graphSize, &linearizedVertices[0]);
	CudaMemory<int> edgesCuda(edges.size(), &edges[0]);
	CudaMemory<unsigned int> costsCuda(graphSize, 0xFF);
	CudaHostMemory<bool> stopCuda;

	// computation
	costsCuda.store(0, 1, source);
	timer.print();

	dim3 blockDim(32, 32);
	int blockCount = (graphSize / (blockDim.x * blockDim.y)) + 1;
	dim3 gridDim(blockCount, 1);

	*stopCuda.host() = false;
	
	timer.start();
	while (!*stopCuda.host())
	{
		*stopCuda.host() = true;
		
		bfsKernel << <gridDim, blockDim >> >(*verticesCuda, *edgesCuda, *costsCuda, graphSize);
		bfsKernelRequeue << <gridDim, blockDim >> >(*verticesCuda, graphSize, stopCuda.device());
	}
	timer.print();

	costs.resize(graphSize, UINT_MAX);
	costsCuda.load(costs[0], graphSize);
}
void bfsSerial(const std::vector<Vertex>& vertices, int source, std::vector<unsigned int>& costs)
{
	if (vertices.size() < 1) return;

	costs.resize(vertices.size(), UINT_MAX);
	std::vector<bool> visited(vertices.size(), false);

	visited[source] = true;
	costs[source] = 0;

	std::queue<int> q;
	q.push(source);

	while (!q.empty())
	{
		int v = q.front();
		q.pop();

		for (int edge : vertices[v].edges)
		{
			if (!visited[edge])
			{
				costs[edge] = costs[v] + 1;
				visited[edge] = true;
				q.push(edge);
			}
		}
	}
}

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

int main()
{
	std::fstream graphFile("init-file.txt", std::ios::in);

	std::vector<Vertex> vertices(1600000, Vertex());

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

	std::vector<unsigned int> costs;
	std::vector<unsigned int> cudaCosts;

	cudaSetDeviceFlags(cudaDeviceMapHost);

	std::cout << "Start BFS" << std::endl;

	Timer timer;
	bfsCuda(vertices, 0, cudaCosts);
	timer.print();

	timer.start();
	bfsSerial(vertices, 0, costs);
	timer.print();

	for (size_t i = 0; i < vertices.size(); i++)
	{
		assert(cudaCosts[i] == costs[i]);
	}

	getchar();

    return 0;
}
