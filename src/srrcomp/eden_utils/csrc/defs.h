/* Copyright 2022 VMware, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>

#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <random>
#include <iomanip>
#include <time.h>
#include <chrono>

using std::cout;
using std::endl;
