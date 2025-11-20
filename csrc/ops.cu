#include "grouped_gemm.h"
#include "grouped_gemm_cublas.h"
#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm_base", &GroupedGemm_base, "Grouped GEMM base.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm_cublas", &GroupedGemm_cuBLAS, "Grouped GEMM cuBLAS.");
}

}  // namespace grouped_gemm
