// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <openvino/op/op.hpp>

namespace llm {
namespace experimental {
class MultiHeadAttention : public ov::op::Op {
  public:
    OPENVINO_OP("MultiHeadAttention", "llm::experimental");

    MultiHeadAttention() = default;
    //
    struct Config {
        int arg_q = 0;                        // query
        int arg_k = 0;                        // will be the same as query if qkv is merged
        int arg_v = 0;                        // will be the same as query if qkv is merged
        int arg_kv_cache = 0;                 // kv cache in shape [2*num_layers, B, H, max_kvLen, S]
        int arg_beam_table = 0;               // beam_table i32[B, max_kvLen] gives item id in batch for each kv-position
        int arg_kv_len = 0;                   // actual kv length i32[1]
        int arg_attn_mask = 0;                // attention mask
        int arg_cos = 0;                      // cos table for RoPE
        int arg_sin = 0;                      // sin table for RoPE

        int rotary_dims = 0;                  //
        int layer_id = 0;                     //
        int n_hidden = 0;
        int n_head = 0;
        int num_kv_heads = 0;
        int rope_type = 0;                    // 0: gptj style, 1: gptneox style
        int multi_query_is_planar = 0;        // chatglm2 is true, others are false
    };
    MultiHeadAttention(const ov::OutputVector &args, Config cfg);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;
    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    bool evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const override;
    bool has_evaluate() const override;

  private:
    Config m_config;
};
} // namespace experimental
} // namespace llm
