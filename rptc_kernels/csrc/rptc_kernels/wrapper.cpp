#include <torch/extension.h>


float decompress(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &out
);


float decompress_matvec_16(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_14(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_12(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_10(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float decompress_matvec_8(
    torch::Tensor &compressed,
    torch::Tensor &codebook,
    torch::Tensor &x,
    torch::Tensor &out
);


float matvec(
    torch::Tensor &decompressed,
    torch::Tensor &x,
    torch::Tensor &out
);


float rowsum(
    torch::Tensor &decompressed,
    torch::Tensor &out
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decompress", &decompress, "decompress");
    m.def("decompress_matvec_16", &decompress_matvec_16, "decompress_matvec_16");
    m.def("decompress_matvec_14", &decompress_matvec_14, "decompress_matvec_14");
    m.def("decompress_matvec_12", &decompress_matvec_12, "decompress_matvec_12");
    m.def("decompress_matvec_10", &decompress_matvec_10, "decompress_matvec_10");
    m.def("decompress_matvec_8", &decompress_matvec_8, "decompress_matvec_8");
    m.def("matvec", &matvec, "matvec");
    m.def("rowsum", &rowsum, "rowsum");
}
