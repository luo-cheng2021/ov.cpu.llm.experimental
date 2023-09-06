// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha.hpp"

namespace llm {
namespace experimental {

MultiHeadAttention::MultiHeadAttention(const ov::OutputVector &args, Config cfg) : Op({args}), m_config(cfg) {
    constructor_validate_and_infer_types();
}

void MultiHeadAttention::validate_and_infer_types() {
    // [B,L,H*S] / [B,L,H*3*S]
    auto qkv_pshape = get_input_partial_shape(m_config.arg_q);

    // output is always [B, L, n_hidden]
    ov::PartialShape output_pshape{qkv_pshape[0], qkv_pshape[1], m_config.n_hidden};

    set_output_type(0, get_input_element_type(0), output_pshape);
}

std::shared_ptr<ov::Node> MultiHeadAttention::clone_with_new_inputs(const ov::OutputVector &new_args) const {
    return std::make_shared<MultiHeadAttention>(new_args, m_config);
}

bool MultiHeadAttention::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("arg_q", m_config.arg_q);
    visitor.on_attribute("arg_k", m_config.arg_k);
    visitor.on_attribute("arg_v", m_config.arg_v);
    visitor.on_attribute("arg_kv_cache", m_config.arg_kv_cache);
    visitor.on_attribute("arg_beam_table", m_config.arg_beam_table);
    visitor.on_attribute("arg_kv_len", m_config.arg_kv_len);
    visitor.on_attribute("arg_attn_mask", m_config.arg_attn_mask);
    visitor.on_attribute("arg_cos", m_config.arg_cos);
    visitor.on_attribute("arg_sin", m_config.arg_sin);

    visitor.on_attribute("rotary_dims", m_config.rotary_dims);
    visitor.on_attribute("layer_id", m_config.layer_id);    
    visitor.on_attribute("n_hidden", m_config.n_hidden);
    visitor.on_attribute("n_head", m_config.n_head);
    return true;
}

bool MultiHeadAttention::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const {
    auto in = inputs[0];
    auto out = outputs[0];
    if (out.data() == in.data()) // Nothing to do
        return true;
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    return true;
}

bool MultiHeadAttention::has_evaluate() const { return false; }
} // namespace experimental
} // namespace llm
