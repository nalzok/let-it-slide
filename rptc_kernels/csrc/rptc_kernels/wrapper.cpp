#include <torch/extension.h>


float decompress(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &out
);


float decompress_matvec_16_8(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_16_7(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_16_6(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_t_16(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_t_14(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decompress", &decompress, "decompress");
    m.def("decompress_matvec_16_8", &decompress_matvec_16_8, "decompress_matvec_16_8");
    m.def("decompress_matvec_16_7", &decompress_matvec_16_7, "decompress_matvec_16_7");
    m.def("decompress_matvec_16_6", &decompress_matvec_16_6, "decompress_matvec_16_6");
    m.def("decompress_matvec_t_16", &decompress_matvec_t_16, "decompress_matvec_t_16");
    m.def("decompress_matvec_t_14", &decompress_matvec_t_14, "decompress_matvec_t_14");
}
