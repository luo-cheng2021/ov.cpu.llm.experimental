// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "d_tensor.hpp"
#include <openvino/op/op.hpp>
#include <string>

namespace llm {
namespace experimental {
class FC : public ov::op::Op {
  public:
    OPENVINO_OP("FC", "llm::experimental");

    FC() = default;

    struct Config {
        std::string quant_type; // Q8_0
        int K = 0;              // raw weight matrix shape
        int N = 0;              // raw weight matrix shape
        int evaluate_qweight = 0;
    };

    FC(const ov::OutputVector &args, Config cfg);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;
    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    bool has_evaluate() const override { return true; }

    bool evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const override;

    // every 32 weights within each output channel share a single scale
    bool quant_Q8_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized) const;
    bool evaluate_Q8_0(d_tensor::PlainTensor<float> x, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> y) const;

    bool quant_Q4_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized) const;
    bool evaluate_Q4_0(d_tensor::PlainTensor<float> x, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> y) const;

    // all weights in each output channel share a single scale (per-OC)
    bool quant_Q8_C(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized,
                    d_tensor::PlainTensor<float> wei_scales) const;
    bool evaluate_Q8_C(d_tensor::PlainTensor<float> x, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<float> wei_scales, d_tensor::PlainTensor<float> y) const;

    bool quant_Q4_C(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized,
                    d_tensor::PlainTensor<int32_t> wei_scales) const;
    bool evaluate_Q4_C(d_tensor::PlainTensor<float> x, d_tensor::PlainTensor<int8_t> wei_quantized,
                       d_tensor::PlainTensor<int32_t> wei_scales, d_tensor::PlainTensor<float> y) const;

    void dynamic_quantize_x(d_tensor::PlainTensor<float>& input, size_t Kgroups, size_t group_k);

  private:
    enum class quantType {
        Unkown,
        Q8_0,
        Q8_C,
        Q4_0,
        Q4_C,
    } m_qtype;

    Config m_config;
    d_tensor::PlainTensor<int8_t> x_quantized{true};
    d_tensor::PlainTensor<float> x_scales{true};
};
} // namespace experimental
} // namespace llm
