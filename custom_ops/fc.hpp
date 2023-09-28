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

    struct Config {
      std::string quant_type;     // Q8_0
      int K = 0;    // raw weight matrix shape 
      int N = 0;    // raw weight matrix shape 
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

    bool quant_q8_0(d_tensor::PlainTensor<float> wei, d_tensor::PlainTensor<int8_t> wei_quantized) const;
    bool evaluate_q8_0(d_tensor::PlainTensor<float> x, d_tensor::PlainTensor<int8_t> wei_quantized, d_tensor::PlainTensor<float> y) const;

  private:

    enum class quantType {
        Unkown,
        Q8_0,
    } m_qtype;

    Config m_config;
    d_tensor::PlainTensor<int8_t> x_quantized{true};
    d_tensor::PlainTensor<float> x_scales{true};
};
} // namespace experimental
} // namespace llm
