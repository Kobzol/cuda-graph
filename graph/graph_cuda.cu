#include "graph_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudamem.h"
#include "util.h"

bool GraphCUDA::CudaInitialized = false;

int GraphCUDA::add_vertex()
{
	int value = Graph::add_vertex();
	this->dirty = true;

	return value;
}
void GraphCUDA::add_edge(int from, int to, unsigned int cost)
{
	Graph::add_edge(from, to, cost);
	this->dirty = true;
}

void GraphCUDA::initCuda()
{
	if (!GraphCUDA::CudaInitialized)
	{
		cudaSetDeviceFlags(cudaDeviceMapHost);
		GraphCUDA::CudaInitialized = true;
	}
}

__global__ void bfsKernel(LinearizedVertex* vertices, Edge* edges, size_t size)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.y + threadIdx.x;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].frontier)
	{
		vertices[pos].frontier = false;

		int edgeCount = vertices[pos].edgeCount;
		int edgeIndex = vertices[pos].edgeIndex;

		for (size_t i = 0; i < edgeCount; i++)
		{
			int edge = edges[edgeIndex + i].target;
			if (!vertices[edge].visited)
			{
				vertices[edge].frontier_next = true;
			}
		}
	}
}
__global__ void bfsRequeueKernel(LinearizedVertex* vertices, size_t size, int target, bool *stop)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.y + threadIdx.x;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].frontier_next)
	{
		vertices[pos].frontier = true;
		vertices[pos].frontier_next = false;
		vertices[pos].visited = true;

		stop[0] = false;

		if (pos == target)
		{
			stop[1] = true;
		}
	}
}
bool GraphCUDA::is_connected(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return false;

	this->relinearizeVertices();
	this->initCuda();

	if (this->edges.size() < 1) return false;

	int graphSize = (int) this->vertices.size();

	this->linearizedVertices[from].frontier = true;
	this->linearizedVertices[from].visited = true;

	CudaMemory<LinearizedVertex> verticesCuda(graphSize, &(this->linearizedVertices[0]));
	CudaMemory<Edge> edgesCuda(this->edges.size(), &(this->edges[0]));
	CudaHostMemory<bool> stopCuda(2);

	// computation
	dim3 blockDim(32, 32);
	int blockCount = (graphSize / (blockDim.x * blockDim.y)) + 1;
	dim3 gridDim(blockCount, 1);

	bool* stopHost = stopCuda.host();
	stopHost[0] = false;

	while (!stopHost[0])
	{
		stopHost[0] = true;

		bfsKernel << <gridDim, blockDim >> >(*verticesCuda, *edgesCuda, graphSize);
		cudaDeviceSynchronize();
		bfsRequeueKernel << <gridDim, blockDim >> >(*verticesCuda, graphSize, to, stopCuda.device());
		cudaDeviceSynchronize();

		if (stopHost[1])
		{
			return true;
		}
	}

	return false;
}

__global__ void dijkstraKernel(LinearizedVertex* vertices, Edge* edges, unsigned int* costs, unsigned int* nextCosts, size_t size)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.y + threadIdx.x;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].frontier)
	{
		vertices[pos].frontier = false;
		unsigned int distance = costs[pos];

		for (size_t i = 0; i < vertices[pos].edgeCount; i++)
		{
			Edge& edge = edges[vertices[pos].edgeIndex + i];
			atomicMin(nextCosts + edge.target, distance + edge.cost);
		}
	}
}
__global__ void dijkstraRequeueKernel(LinearizedVertex* vertices, unsigned int* costs, unsigned int* nextCosts, size_t size, bool *stop)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.y + threadIdx.x;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (nextCosts[pos] < costs[pos])
	{
		vertices[pos].frontier = true;
		costs[pos] = nextCosts[pos];
		*stop = false;
	}

	nextCosts[pos] = costs[pos];
	
}
unsigned int GraphCUDA::get_shortest_path(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return UINT_MAX;

	this->relinearizeVertices();
	this->initCuda();

	if (this->edges.size() < 1) return UINT_MAX;

	int graphSize = (int) this->vertices.size();

	this->linearizedVertices[from].frontier = true;

	CudaMemory<LinearizedVertex> verticesCuda(graphSize, &(this->linearizedVertices[0]));
	CudaMemory<Edge> edgesCuda(this->edges.size(), &(this->edges[0]));

	std::vector<unsigned int> costs(graphSize, UINT_MAX);
	CudaMemory<unsigned int> costsCuda(graphSize, 0xFF);
	CudaMemory<unsigned int> nextCostsCuda(graphSize, 0xFF);
	
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

		dijkstraKernel << <gridDim, blockDim >> >(*verticesCuda, *edgesCuda, *costsCuda, *nextCostsCuda, graphSize);
		cudaDeviceSynchronize();
		dijkstraRequeueKernel << <gridDim, blockDim >> >(*verticesCuda, *costsCuda, *nextCostsCuda, graphSize, stopCuda.device());
		cudaDeviceSynchronize();
	}

	costsCuda.load(costs[0], graphSize);

	return costs[to];
}

void GraphCUDA::relinearizeVertices()
{
	this->edges.clear();
	this->linearizedVertices.clear();

	for (const Vertex& vertex : this->vertices)
	{
		int edgeCount = (int) vertex.edges.size();
		int edgeIndex = (int) edges.size();

		this->edges.insert(this->edges.end(), vertex.edges.begin(), vertex.edges.end());
		this->linearizedVertices.emplace_back(edgeIndex, edgeCount);
	}

	this->dirty = false;
}