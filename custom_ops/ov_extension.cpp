// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "mha.hpp"
#include "fc.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({

    // Register operation itself, required to be read from IR
    std::make_shared<ov::OpExtension<llm::experimental::MultiHeadAttention>>(),

    // Register operaton mapping, required when converted from framework model format
    std::make_shared<ov::frontend::OpExtension<llm::experimental::MultiHeadAttention>>(),
    
    // Register operation itself, required to be read from IR
    std::make_shared<ov::OpExtension<llm::experimental::FC>>(),

    // Register operaton mapping, required when converted from framework model format
    std::make_shared<ov::frontend::OpExtension<llm::experimental::FC>>()
}));
