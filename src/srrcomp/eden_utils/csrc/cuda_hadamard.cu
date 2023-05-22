/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/* @author: Shay Vargaftik (VMware Research) */

/* 
 * Inspired by CUDA samples https://docs.nvidia.com/cuda/cuda-samples/index.html (see notice below). 
 * 
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "cuda_hadamard.h"

__global__ void HadamardSharedMemoryIterations(float* vec, unsigned int iters)
{
	// see https://developer.nvidia.com/blog/cooperative-groups/
	namespace cg = cooperative_groups;

	// handle to thread block
	cg::thread_block block = cg::this_thread_block();

	// block's shared memory
	extern __shared__ float shared_memory[];

	// each block perform fwht to a chuck of size n using n / blockDim.x threads
	unsigned int n = 1 << iters;

	// This is the offset of the current block's chunk
	float* block_vec = vec + (blockIdx.x << iters);

	// copy block values to shared memory - each thread copies n / blockDim.x values
	for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
	{
		shared_memory[i] = block_vec[i];
	}

	// initial stride size
	unsigned int stride = 1;

	// requires radix2 step
	if (iters % 2 != 0) {

		// make sure all block values are available in shared memory 
		cg::sync(block);

		for (unsigned int h = threadIdx.x; h < (n >> 1); h += blockDim.x)
		{
			unsigned int index_a = h << 1;
			unsigned int index_b = index_a + 1;

			float a = shared_memory[index_a];
			float b = shared_memory[index_b];

			shared_memory[index_a] = a + b;
			shared_memory[index_b] = a - b;
		}

		stride <<= 1;
	}

	// the rest are radix4 steps
	for (; stride <= (n >> 2); stride <<= 2)
	{
		for (unsigned int h = threadIdx.x; h < (n >> 2); h += blockDim.x)
		{
			unsigned int offset = h & (stride - 1);

			unsigned int index_a = ((h - offset) << 2) + offset;
			unsigned int index_b = index_a + stride;
			unsigned int index_c = index_b + stride;
			unsigned int index_d = index_c + stride;

			// make sure all block threads' updated values are available in shared memory 
			cg::sync(block);

			float a = shared_memory[index_a];
			float b = shared_memory[index_b];
			float c = shared_memory[index_c];
			float d = shared_memory[index_d];

			// radix 2 for [a,b]
			float temp1 = a + b;
			float temp2 = a - b;

			// radix 2 for [c,d]
			float temp3 = c + d;
			float temp4 = c - d;

			// radix 2 for [a,b] and [c,d]
			shared_memory[index_a] = temp1 + temp3;
			shared_memory[index_b] = temp2 + temp4;
			shared_memory[index_c] = temp1 - temp3;
			shared_memory[index_d] = temp2 - temp4;
		}
	}

	// all block threads' values are available in shared memory 
	cg::sync(block);

	// copy values from shared memory
	for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
	{
		block_vec[i] = shared_memory[i];
	}
}

__global__ void HadamardRadix2Iteration(float* vec, unsigned int stride)
{
	unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int index_a = (h << 1) - (h & (stride - 1));
	unsigned int index_b = index_a + stride;

	float a = vec[index_a];
	float b = vec[index_b];

	vec[index_a] = a + b;
	vec[index_b] = a - b;
}

__global__ void HadamardRadix4Iteration(float* vec, unsigned int stride)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// same as index % stride
	unsigned int offset = index & (stride - 1);

	unsigned int index_a = ((index - offset) << 2) + offset;
	unsigned int index_b = index_a + stride;
	unsigned int index_c = index_b + stride;
	unsigned int index_d = index_c + stride;

	float a = vec[index_a];
	float b = vec[index_b];
	float c = vec[index_c];
	float d = vec[index_d];

	// radix 2 for [a,b]
	float temp1 = a + b;
	float temp2 = a - b;

	// radix 2 for [c,d]
	float temp3 = c + d;
	float temp4 = c - d;

	// radix 2 for [a,b] and [c,d]
	vec[index_a] = temp1 + temp3;
	vec[index_b] = temp2 + temp4;
	vec[index_c] = temp1 - temp3;
	vec[index_d] = temp2 - temp4;
}

__global__ void HadamardRadix8Iteration(float* vec, unsigned int stride)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// same as index % stride
	unsigned int offset = index & (stride - 1);

	unsigned int index_a = ((index - offset) << 3) + offset;
	unsigned int index_b = index_a + stride;
	unsigned int index_c = index_b + stride;
	unsigned int index_d = index_c + stride;
	unsigned int index_e = index_d + stride;
	unsigned int index_f = index_e + stride;
	unsigned int index_g = index_f + stride;
	unsigned int index_h = index_g + stride;

	// radix 4 for [a,b,c,d]
	float a = vec[index_a];
	float b = vec[index_b];
	float c = vec[index_c];
	float d = vec[index_d];

	float temp1 = a + b;
	float temp2 = a - b;
	float temp3 = c + d;
	float temp4 = c - d;

	a = temp1 + temp3;
	b = temp2 + temp4;
	c = temp1 - temp3;
	d = temp2 - temp4;

	// radix 4 for [e,f,g,h]
	float e = vec[index_e];
	float f = vec[index_f];
	float g = vec[index_g];
	float h = vec[index_h];

	temp1 = e + f;
	temp2 = e - f;
	temp3 = g + h;
	temp4 = g - h;

	e = temp1 + temp3;
	f = temp2 + temp4;
	g = temp1 - temp3;
	h = temp2 - temp4;

	// radix 2 for [a,b,c,d] and [e,f,g,h]
	vec[index_a] = a + e;
	vec[index_b] = b + f;
	vec[index_c] = c + g;
	vec[index_d] = d + h;
	vec[index_e] = a - e;
	vec[index_f] = b - f;
	vec[index_g] = c - g;
	vec[index_h] = d - h;
}

void HadamardWithCudaNoSharedMemory(float* vec, int n, int device)
{
	if ((n & (n - 1)) != 0)
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCudaNoSharedMemory failed. Input size is not a power of 2 ***\n");
		return;
	}
	
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCudaNoSharedMemory failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}
	
	int maxThreadsPerBlock;

	// read the following attribute from the cuda device
	cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);

	int radix2_num_threads = (n >> 1) < maxThreadsPerBlock ? (n >> 1) : maxThreadsPerBlock;
	int radix4_num_threads = (n >> 2) < maxThreadsPerBlock ? (n >> 2) : maxThreadsPerBlock;
	int radix8_num_threads = (n >> 3) < maxThreadsPerBlock ? (n >> 3) : maxThreadsPerBlock;

	int log2n = (int)log2(n);
	int len = 1;

	if ((log2n % 3) % 2 != 0) 
	{
		HadamardRadix2Iteration <<< (n >> 1) / radix2_num_threads, radix2_num_threads >>> (vec, len);
		len <<= 1;
		log2n -= 1;
	}

	if (log2n % 3 != 0) 
	{
		HadamardRadix4Iteration <<< (n >> 2) / radix4_num_threads, radix4_num_threads >>> (vec, len);
		len <<= 2;
	}

	for (; len < n; len <<= 3) 
	{
		HadamardRadix8Iteration <<< (n >> 3) / radix8_num_threads, radix8_num_threads >>> (vec, len);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCudaNoSharedMemory failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}
}

void HadamardWithCuda(float* vec, int n, int device)
{
	if ((n & (n - 1)) != 0)
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. Input size is not a power of 2 ***\n");
		return;
	}
	
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaGetErrorString(cudaStatus));
	}

	int maxThreadsPerBlock;
	int sharedMemPerBlock;

	// read the following attributes from the cuda device
	cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device);
	cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device);

	int log2n = (int)log2(n);

	// constraint on block's shared memory size
	int sharedMemPerBlockfFloats = sharedMemPerBlock / sizeof(float);
	int log2nSharedMemPerBlockfFloats = (int)log2(sharedMemPerBlockfFloats);
	int sharedMemIters = log2nSharedMemPerBlockfFloats > log2n ? log2n : log2nSharedMemPerBlockfFloats;

	// must ensure that only radix8 iterations remain after shared memory iterations
	if ((log2n - sharedMemIters) % 3 == 2) 
	{
		sharedMemIters -= 1;
	}
	else if ((log2n - sharedMemIters) % 3 == 1) 
	{
		sharedMemIters -= 2;
	}

	// the shared memory allocated size per block
	int sharedMemSize = 1 << sharedMemIters;

	// the number of transformed chunks after sm iterations
	int num_blocks = 1 << (log2n - sharedMemIters);

	// number of threads per block
	int num_threads;

	if (sharedMemSize == 2) 
	{
		num_threads = 1;
	}
	else 
	{
		num_threads = (sharedMemSize >> 2) < maxThreadsPerBlock ? (sharedMemSize >> 2) : maxThreadsPerBlock;
	}

	HadamardSharedMemoryIterations <<< num_blocks, num_threads, sharedMemSize * sizeof(float) >>> (vec, sharedMemIters);

	int radix8_num_threads = (n >> 3) < maxThreadsPerBlock ? (n >> 3) : maxThreadsPerBlock;

	// complete the transform	
	for (int len = sharedMemSize; len < n; len <<= 3) 
	{
		HadamardRadix8Iteration <<< (n >> 3) / radix8_num_threads, radix8_num_threads >>> (vec, len);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "\n*** (CPP) HadamardWithCuda failed. %s ***\n", cudaGetErrorString(cudaGetLastError()));
	}
}
