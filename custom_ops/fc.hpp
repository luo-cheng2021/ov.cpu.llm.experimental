// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <openvino/op/op.hpp>
#include "d_tensor.hpp"

namespace llm {
namespace experimental {
class FC : public ov::op::Op {
  public:
    OPENVINO_OP("FC", "llm::experimental");

    FC() = default;
    //
    struct Config {
      int quant_type = 2;         // ['', 'nncf_w8', 'llama_q8w8_0']
      int llama_quant_type = 3;   // ['', 'tensor', 'channel', 'group']
      int llama_group_k = 32;
      int llama_group_n = 32;
      // raw weight matrix size is known (K x N)
      int K = 0;
      int N = 0;
      int bits = 8;
      int evaluate_qweight = 0;
    };
    FC(const ov::OutputVector &args, Config cfg);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;
    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    bool has_evaluate() const override {
        return true;
    }
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;

    bool quant_q8_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized, d_tensor::PlainTensor<float> wei_scales) const;
    bool evaluate_q8_0(d_tensor::PlainTensor<float> x, d_tensor::PlainTensor<int8_t> wei_quantized, d_tensor::PlainTensor<float> wei_scales, d_tensor::PlainTensor<float> y) const;

  private:
    Config m_config;
    d_tensor::PlainTensor<int8_t> x_quantized{true};
    d_tensor::PlainTensor<float> x_scales{true};
};
} // namespace experimental
} // namespace llm
