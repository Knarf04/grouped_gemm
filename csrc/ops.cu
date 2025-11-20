#include "grouped_gemm.h"
#include "grouped_gemm_cublas.h"
#include "grouped_gemm_cutlass.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm_base", &GroupedGemm_base, "Grouped GEMM base.");
  m.def("gmm_cuBLAS", &GroupedGemm_cuBLAS, "Grouped GEMM cuBLAS.");
  m.def("gmm_CUTLASS", &GroupedGemm_CUTLASS, "Grouped GEMM CUTLASS.");
}

}  // namespace grouped_gemm
