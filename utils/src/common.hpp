#include <pybind11/pybind11.h>

template<typename T, int RANK>
struct tensor {
    pybind11::buffer_info info;

    tensor(pybind11::buffer& b) {
        info = b.request();
        if (sizeof(T) != info.itemsize) {
            throw std::invalid_argument("tensor data type mismatch with pybind11::buffer");
        }
        if (RANK != info.ndim) {
            throw std::invalid_argument("tensor rank mismatch with pybind11::buffer");
        }
    }
    template<int idim>
    size_t get_offset(int coordinate) {
        return coordinate * info.strides[idim];
    }

    template<int idim, typename... Args>
    size_t get_offset(int coordinate, Args... args) {
        size_t offset = coordinate * info.strides[idim];
        return offset + get_offset<idim + 1>(args...);
    }

    template<typename... Args>
    T& operator()(Args... args) {
        assert(sizeof...(args) <= RANK);
        return *reinterpret_cast<T*>(reinterpret_cast<int8_t*>(info.ptr) + get_offset<0>(args...));
    }

    pybind11::ssize_t size(int idim) {
        return info.shape[idim];
    }
};
