#include <torch/extension.h>
#include <cuSZp_entry_f32.h>
#include <cuSZp_timer.h>
#include <cuSZp_utility.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)



torch::Tensor compress(torch::Tensor input, float error_bound,
                       std::string mode) {
  CHECK_INPUT(input);
  // Get the input tensor's data pointer and size
  float *d_input_data = input.data_ptr<float>();
  int64_t num_elements = input.numel();
  size_t compressed_size = 0;

  // Cuda allocate memory for the compressed output
  unsigned char *d_compressed_data;
  cudaMalloc((void **)&d_compressed_data, num_elements * sizeof(float));
  cudaMemset(d_compressed_data, 0, num_elements * sizeof(float));

  // Initializing CUDA Stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Just a warmup.
  SZp_compress_deviceptr_f32(d_input_data, d_compressed_data, num_elements,
                             &compressed_size, error_bound, stream);
  // Create a new tensor on the GPU from the compressed output
  cudaStreamSynchronize(stream);
  torch::Tensor output = torch::empty(
      {compressed_size}, torch::TensorOptions()
                             .dtype(torch::kUInt8)
                             .device(torch::kCUDA)
                             .memory_format(torch::MemoryFormat::Contiguous));
  // write from d_compressed_data
  cudaMemcpy(output.data_ptr<unsigned char>(), d_compressed_data,
             compressed_size, cudaMemcpyDeviceToDevice);
  // Sync free
  cudaStreamSynchronize(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }


  cudaFree(d_compressed_data);
  cudaStreamDestroy(stream);
  CHECK_INPUT(output);
  return output;
}

torch::Tensor decompress(torch::Tensor compressed_data, int64_t num_elements,
                         size_t compressed_size, float error_bound,
                         std::string mode) {
  CHECK_INPUT(compressed_data);
  // Get the input tensor's data pointer and size
  unsigned char *d_compressed_data = compressed_data.data_ptr<unsigned char>();

  torch::Tensor decompressed_data = torch::zeros(
      {num_elements}, torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(torch::kCUDA)
                          .memory_format(torch::MemoryFormat::Contiguous));
  float *d_decompressed_data = decompressed_data.data_ptr<float>();

  // Initializing CUDA Stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  SZp_decompress_deviceptr_f32(d_decompressed_data, d_compressed_data,
                               num_elements, compressed_size, error_bound,
                               stream);
  cudaStreamSynchronize(stream);
  // Check cuda errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaStreamDestroy(stream);
  CHECK_INPUT(decompressed_data);
  return decompressed_data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress", &compress, "Compress a PyTorch tensor using cuSZp");
  m.def("decompress", &decompress, "Decompress a PyTorch tensor using cuSZp");
}
