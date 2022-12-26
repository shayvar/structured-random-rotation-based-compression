// Copyright 2022 VMware, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/* @author: Shay Vargaftik (VMware Research) */

#include <torch/extension.h>
#include "./csrc/cuda_hadamard.h"
#include "./csrc/cuda_packing.h"


using namespace std;
using namespace torch::indexing;


torch::Tensor Hadamard(torch::Tensor vec)
{
	TORCH_CHECK(vec.device().type() == torch::kCUDA, "input must be a CUDA tensor");
	TORCH_CHECK(vec.dtype() == torch::kFloat32, "input must be a torch.float32 CUDA tensor");

	// size of last dimension
	auto n = vec.size(-1);

	TORCH_CHECK(n == vec.numel(), "input must be 1D");
	TORCH_CHECK((n & (n - 1)) == 0 && n > 0, "input size must be a power of 2");

	// cloning makes the output vector contiguous
	auto output = vec.clone(); 

	// device number
	int device = output.device().index();
	
	// invoke Hadamard kernel
	HadamardWithCuda(output.data_ptr<float>(), n, device);

	return output;
}


torch::Tensor EdenBinsToBits(torch::Tensor bins, int nbits)
{
	TORCH_CHECK(bins.device().type() == torch::kCUDA, "bins must be a CUDA tensor");
	TORCH_CHECK(bins.dtype() == torch::kInt32, "bins must be a torch.int32 CUDA tensor");

	// size of last dimension
	auto nbins = bins.size(-1);

	TORCH_CHECK(nbins == bins.numel(), "bins must be 1D");
	TORCH_CHECK((nbins & 31) == 0, "bins size must be a multiple of 32");

	// cloning makes the tensor contiguous
	auto cbins = bins.clone();  
	
	// device number
	int device = cbins.device().index();

	// allocate tensor for packed bits
	auto carr_options = torch::TensorOptions().device(bins.device().type(), device).dtype(torch::kInt32);
	torch::Tensor carr = torch::zeros({(nbins>>5) * nbits}, carr_options);
	
	// call cuda kernel
	BinsToBits((int *)cbins.data_ptr(), nbins, (uint32_t *)carr.data_ptr(), (nbins>>5) * nbits, nbits, device);  

	return carr;
}


torch::Tensor EdenBitsToBins(torch::Tensor arr, int nbits)
{
	TORCH_CHECK(arr.device().type() == torch::kCUDA, "arr must be a CUDA tensor");
	TORCH_CHECK(arr.dtype() == torch::kInt32, "arr must be a torch.int32 CUDA tensor");

	// size of last dimension
	auto narr = arr.size(-1);

	TORCH_CHECK(narr == arr.numel(), "arr must be 1D");

	// cloning makes the tensor contiguous
	auto carr = arr.clone();  

	// device number
	int device = carr.device().index();
	
	// allocate tensor for unpacked bins
	auto cbins_options = torch::TensorOptions().device(arr.device().type(), device).dtype(torch::kInt32);
	torch::Tensor cbins = torch::zeros({(narr / nbits) << 5}, cbins_options);
	
	// call cuda kernel
	BitsToBins((int *)cbins.data_ptr(), (narr / nbits) << 5, (uint32_t *)carr.data_ptr(), narr, nbits, device);  

	return cbins;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, modul) {
	modul.def("Hadamard", &Hadamard, "fast Hadamard transform");
	modul.def("EdenBinsToBits", &EdenBinsToBits, "EDEN's packing bins to bits");
	modul.def("EdenBitsToBins", &EdenBitsToBins, "EDEN's unpacking bits to bins");
}

