#include <torch/extension.h>


void decompress(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &out
);


void decompress_matvec_16(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


void decompress_matvec_14(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


void decompress_matvec_12(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


void decompress_matvec_10(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


void decompress_matvec_8(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decompress", &decompress, "decompress");
    m.def("decompress_matvec_16", &decompress_matvec_16, "decompress_matvec_16");
    m.def("decompress_matvec_14", &decompress_matvec_14, "decompress_matvec_14");
    m.def("decompress_matvec_12", &decompress_matvec_12, "decompress_matvec_12");
    m.def("decompress_matvec_10", &decompress_matvec_10, "decompress_matvec_10");
    m.def("decompress_matvec_8", &decompress_matvec_8, "decompress_matvec_8");
}
