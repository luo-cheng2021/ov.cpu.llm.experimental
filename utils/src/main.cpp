#include <pybind11/pybind11.h>
#include "common.hpp"

namespace py = pybind11;

void update_beam_table(py::buffer _beam_idx, py::buffer _beam_table, int beam_length) {
    tensor<int32_t, 2> beam_idx(_beam_idx);
    tensor<int32_t, 2> beam_table(_beam_table);

    auto beam_size = beam_idx.size(0);
    for (size_t beam = 0; beam < beam_size; beam++) {
        // trace back along the beam
        int32_t cur_b = beam;
        for (int p = beam_length - 1; p >=0; p--) {
            beam_table(beam, p) = cur_b;
            cur_b = beam_idx(cur_b, p);
        }
    }
}


PYBIND11_MODULE(utils_cpp, m) {
    m.def("update_beam_table", &update_beam_table);
}
