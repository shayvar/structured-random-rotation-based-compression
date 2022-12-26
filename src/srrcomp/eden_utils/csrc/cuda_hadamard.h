/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef CUDA_HADAMARD
#define CUDA_HADAMARD

#include "defs.h"

void HadamardWithCudaNoSharedMemory(float* vec, int n, int device);
void HadamardWithCuda(float* vec, int n, int device);

#endif