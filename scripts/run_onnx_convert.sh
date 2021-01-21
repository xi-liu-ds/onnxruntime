#!/bin/bash

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# Check if we have enough arguments.
if [ "$#" -ne 2 ] && [ "$#" -ne 3 ] ; then
    echo "run_onnx_convert.sh <model_root_path> <model_class> <test_file_path> -- requires 2 arguments (model_root_path & model_class), 1 optional (test_file_path)"
    echo "  Note: <model_class> can be 'GPT2LMHeadModel', 'GPT2LMHeadModel_NoPadding' or 'GPT2LMHeadModel_BeamSearchStep'"
    exit
fi

echo "Processing models at $1 to generate $2 onnx model"

python /xiaoyudisk/xiaoyu/onnxruntime/onnxruntime/python/tools/transformers/convert_to_onnx.py \
--model_name_or_path=$1 --model_class=$2 --output=$1/gpt2_beamsearchstep_int8.onnx -o --precision=int8 \
--input_test_file=$3