#include "graph_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudamem.h"

__global__ void bfsKernel(LinearizedVertex* vertices, int* edges, unsigned int* costs, size_t size)
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
__global__ void bfsKernelRequeue(LinearizedVertex* vertices, size_t size, bool *stop)
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

void cudaInit()
{
	cudaSetDeviceFlags(cudaDeviceMapHost);
}

int GraphCUDA::add_vertex()
{
	int value = Graph::add_vertex();
	this->dirty = true;

	return value;
}
void GraphCUDA::add_edge(int from, int to)
{
	Graph::add_edge(from, to);
	this->dirty = true;
}

bool GraphCUDA::is_connected(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return false;

	this->relinearizeVertices();

	if (this->edges.size() < 1) return false;

	int graphSize = (int) this->vertices.size();

	linearizedVertices[from].frontier = true;
	linearizedVertices[from].visited = true;

	CudaMemory<LinearizedVertex> verticesCuda(graphSize, &(this->linearizedVertices[0]));
	CudaMemory<int> edgesCuda(this->edges.size(), &(this->edges[0]));
	CudaMemory<unsigned int> costsCuda(graphSize, 0xFF);
	CudaHostMemory<bool> stopCuda;

	// computation
	costsCuda.store(0, 1, from);

	dim3 blockDim(32, 32);
	int blockCount = (graphSize / (blockDim.x * blockDim.y)) + 1;
	dim3 gridDim(blockCount, 1);

	bool* stopHost = stopCuda.host();
	*stopHost = false;

	while (!(*stopHost))
	{
		*stopHost = true;

		bfsKernel << <gridDim, blockDim >> >(*verticesCuda, *edgesCuda, *costsCuda, graphSize);
		bfsKernelRequeue << <gridDim, blockDim >> >(*verticesCuda, graphSize, stopCuda.device());
	}

	std::vector<unsigned int> costs(graphSize, UINT_MAX);
	costsCuda.load(costs[0], graphSize);

	return costs[to] != UINT_MAX;
}

void GraphCUDA::relinearizeVertices()
{
	this->edges.clear();
	this->linearizedVertices.clear();

	for (const Vertex& vertex : this->vertices)
	{
		int edgeCount = (int) vertex.edges.size();
		int edgeIndex = (int) edges.size();

		for (int edge : vertex.edges)
		{
			this->edges.push_back(edge);
		}

		this->linearizedVertices.emplace_back(edgeIndex, edgeCount);
	}

	this->dirty = false;
}