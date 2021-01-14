# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script helps onnx conversion and validation for GPT2 model with past state.
import os
import logging
import torch
import onnx
import random
import numpy
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from benchmark_helper import Precision

logger = logging.getLogger(__name__)

PRETRAINED_GPT2_MODELS = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

DEFAULT_TOLERANCE = {Precision.FLOAT32: 0.0005, Precision.FLOAT16: 0.2, Precision.INT8: 3.0}

class MyGPT2LMHeadModel(GPT2LMHeadModel):
    """ Here we wrap a class for Onnx model conversion for GPT2LMHeadModel with past state.
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, position_ids, attention_mask, beam_select_idx, *past):
        past = [past[i].index_select(1,beam_select_idx[0]) for i in range(len(past))]
        logits, presents = super().forward(input_ids, position_ids=position_ids, attention_mask=attention_mask, past=past)       
        return torch.nn.functional.log_softmax(logits[: , -1 , :], dim=-1), presents


class MyGPT2LMHeadModel_NoPadding(GPT2LMHeadModel):
    """ Here we wrap a class for Onnx model conversion for GPT2LMHeadModel with past state and no padding.
        When you always use batch_size=1 in inference, there is no padding in inputs. In such case, position_ids
        and attention_mask need no be in inputs.
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, beam_select_idx, *past):
        past =[[past[i][j][beam_select_idx[0], ...] for j in range(len(past[0]))] for i in range(len(past))]
        logits, presents = super().forward(input_ids, past=past)       
        return torch.nn.functional.log_softmax(logits[: , -1 , :], dim=-1), presents

class MyGPT2LMHeadModel_BeamSearchStep(GPT2LMHeadModel):
    '''
    This is one step beam search for GPT2.
    
    Note that: 
    1. This is a batch beam search, meaning, it does beam search for multiple input sequence at once, which should increase throughput.
    2. If some sequences are finished (has EOS token at the end), it will still go into model call, which might be expensive if all the sequences are 
        exepected to have very difference lengths. 
        To solve this issue, one can do the following steps
        a.) 'torch.gather` to get the unfinished sequences/sents
        b.) call model on the unfinished sequences only
        c.) use `torch.scatter` to insert thef finished sequences back, and do the beam search step.
    '''
    def __init__(self, config, beam_size=4):
        super().__init__(config)
        self.beam_size = beam_size

    def forward(
        self, 
        input_ids,
        position_ids, 
        attention_mask, 
        beam_select_idx,
        input_log_probs, 
        input_unfinished_sents,  
        *past, 
    ):
        '''
        param input_ids: 3D, shape=(batch, beam_size/1, seq_len). If your input is 2D, insert a dimesion like this `input_ids = input_ids.unsqueeze(1)`
        param log_probs: shape=(batch, beam_size)
        param unfinished_sents: shape=(batch, beam_size), each cell indicates if a sequence is finished or not (i.e., having EOS at the end.)
        '''
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(batch_size, -1, input_ids.size(-1))
        past = [past[i].index_select(1,beam_select_idx[0]) for i in range(len(past))]
        logits_flat, present_flat = super().forward(
            input_ids, 
            position_ids=position_ids, 
            attention_mask=attention_mask,
            past=past
        )   
        logits_flat = torch.nn.functional.log_softmax(logits_flat[: , -1 , :], dim=-1)
        next_token_logits = logits_flat[:, -1].view(batch_size, -1, logits_flat.size(-1))
        next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
        next_token_log_probs, next_token_ids = torch.topk(next_token_log_probs, self.beam_size, dim=-1, largest=True, sorted=True)

        # finished sentences is always with EOS, and all but the first one has -inf, so that they will be automatically dropped in the round of beam search.
        finished_sents = ~input_unfinished_sents
        next_token_log_probs.masked_fill_(finished_sents.unsqueeze(-1), -numpy.inf)
        next_token_log_probs[..., 0].masked_fill_(finished_sents, 0)
        next_token_ids.masked_fill_(finished_sents.unsqueeze(-1), self.config.eos_token_id)

        output_log_probs = input_log_probs.unsqueeze(-1) + next_token_log_probs

        # select N sequences from beams of each input, sorted by sequence probability
        output_log_probs = output_log_probs.view(batch_size, -1)  # shape=(batch, beam_size^2)
        output_log_probs, selected_index_flat = output_log_probs.topk(self.beam_size, dim=-1, largest=True, sorted=True)  # output shape=(batch, beam_size)

        # select the correspondent sentences/next tokens
        selected_input_seq = selected_index_flat // self.beam_size
        next_token_ids = next_token_ids.view(batch_size, -1).gather(-1, selected_index_flat)

        input_ids = input_ids.gather(1, selected_input_seq.unsqueeze(-1).repeat(1, 1, input_ids.size(-1)))
        # NOTE: to handle hidden state cache,  you'll need `selected_input_seq` to select the proper `present`/`past` state, the way as `input_ids` is selected above.
        # for i in range(len(present_flat)):
        #   print(present_flat[i].shape, selected_input_seq.unsqueeze(-1).repeat(1, 1, present_flat[i].size(-1)).shape)
        #   present_flat[i] = present_flat[i].gather(1, selected_input_seq.unsqueeze(-1).repeat(1, 1, present_flat[i].size(-1)))

        output_unfinished_sents = input_unfinished_sents.gather(1, selected_input_seq)
        output_unfinished_sents = output_unfinished_sents & next_token_ids.ne(self.config.eos_token_id)

        # get the next full input_ids
        input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(-1)], dim=-1).contiguous()

        return input_ids, present_flat, selected_index_flat, output_log_probs, output_unfinished_sents

# Maps model class name to a tuple of model class, name of first output and use padding or not
MODEL_CLASSES = {
    'GPT2LMHeadModel': (MyGPT2LMHeadModel, 'logits', True),
    'GPT2LMHeadModel_NoPadding': (MyGPT2LMHeadModel_NoPadding, 'logits', False),
    'GPT2LMHeadModel_BeamSearchStep': (MyGPT2LMHeadModel_BeamSearchStep, 'logits', True),
    #'GPT2Model': (MyGPT2Model, 'last_state', True),
}


class Gpt2Inputs:
    def __init__(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past, 
        beam_select_idx=None, 
        input_log_probs=None, 
        input_unfinished_sents=None
    ):
        self.input_ids: torch.LongTensor = input_ids
        self.position_ids: torch.LongTensor = position_ids
        self.attention_mask: Union[torch.FloatTensor, torch.HalfTensor] = attention_mask
        self.past: Union[List[torch.FloatTensor], List[torch.HalfTensor]] = past
        if beam_select_idx is None:
            self.beam_select_idx: torch.LongTensor = torch.zeros([1,len(input_ids)]).long()
        else:
            self.beam_select_idx: torch.LongTensor = beam_select_idx
        self.input_log_probs: torch.FloatTensor = input_log_probs      
        self.input_unfinished_sents: torch.ByteTensor = input_unfinished_sents

    def to_list(self) -> List:
        input_list = [v for v in ([
            self.input_ids, 
            self.position_ids, 
            self.attention_mask, 
            self.beam_select_idx,
            self.input_log_probs, 
            self.input_unfinished_sents
        ]) if v is not None]
        if self.past:
            input_list.extend(self.past)
        
        return input_list

    def to_kwargs(self):
        return self.input_ids, self.position_ids, self.attention_mask, self.beam_select_idx, self.past, self.input_log_probs, self.input_unfinished_sents

    def to_tuple(self) -> Tuple:
        return tuple(v for v in [self.input_ids, self.position_ids, self.attention_mask, self.beam_select_idx, self.past, self.input_log_probs, self.input_unfinished_sents] if v is not None)

    def to_fp32(self):
        attention_mask = self.attention_mask.to(dtype=torch.float32) if self.attention_mask is not None else None
        past = [p.to(dtype=torch.float32) for p in self.past]
        return Gpt2Inputs(
            self.input_ids, 
            self.position_ids, 
            attention_mask, 
            past,
            self.beam_select_idx,
            self.input_log_probs,
            self.input_unfinished_sents
        )


class Gpt2Helper:
    """ A helper class for Gpt2 model conversion, inference and verification.
    """
    @staticmethod
    def get_dummy_inputs(batch_size: int,
                         past_sequence_length: int,
                         sequence_length: int,
                         num_attention_heads: int,
                         hidden_size: int,
                         num_layer: int,
                         vocab_size: int,
                         device: torch.device,
                         float16: bool = False,
                         has_position_ids: bool = True,
                         has_attention_mask: bool = True,
                         has_beam_select_idx: bool = True,
                         has_beam_search: bool = True) -> Gpt2Inputs:
        """ Create random inputs for GPT2 model.
        Returns torch tensors of input_ids, position_ids, attention_mask and a list of past state tensors.
        """
        float_type = torch.float16 if float16 else torch.float32
        past_shape = [2, batch_size, num_attention_heads, past_sequence_length, int(hidden_size / num_attention_heads)]

        past = [torch.rand(past_shape, dtype=float_type, device=device) for _ in range(num_layer)]
        input_ids = torch.randint(low=0,
                                  high=vocab_size - 1,
                                  size=(batch_size, sequence_length),
                                  dtype=torch.int64,
                                  device=device)

        attention_mask = None
        if has_attention_mask:
            total_sequence_length = past_sequence_length + sequence_length
            attention_mask = torch.ones([batch_size, total_sequence_length], dtype=float_type, device=device)
            if total_sequence_length >= 2:
                padding_position = random.randint(0, total_sequence_length - 1)  # test input with padding.
                attention_mask[:, padding_position] = 0

        # Deduce position_ids from attention mask
        beam_select_idx = None
        if has_beam_select_idx:
            beam_select_idx = torch.zeros([1,batch_size]).long()
        
        input_log_probs = None
        input_unfinished_sents = None
        if has_beam_search:
            input_log_probs = torch.zeros([batch_size, 1], dtype=float_type, device=device)
            input_unfinished_sents = torch.ones([batch_size, 1], dtype=torch.bool, device=device)

        # Deduce position_ids from attention mask
        position_ids = None
        if has_position_ids:
            position_ids = (attention_mask.long().cumsum(-1) - 1)
            position_ids.masked_fill_(position_ids < 0, 0)
            position_ids = position_ids[:, past_sequence_length:]

        return Gpt2Inputs(
            input_ids, 
            position_ids, 
            attention_mask, 
            past, 
            beam_select_idx,
            input_log_probs,
            input_unfinished_sents
        )

    @staticmethod
    def get_output_shapes(batch_size: int,
                          past_sequence_length: int,
                          sequence_length: int,
                          config: GPT2Config,
                          model_class: str = "GPT2LMHeadModel") -> Dict[str, List[int]]:
        """ Returns a dictionary with output name as key, and shape as value.
        """
        num_attention_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        num_layer = config.num_hidden_layers
        vocab_size = config.vocab_size

        output_name = MODEL_CLASSES[model_class][1]

        last_state_shape = [batch_size, sequence_length, vocab_size if output_name == "logits" else hidden_size]
        present_state_shape = [
            2, batch_size, num_attention_heads, past_sequence_length + sequence_length,
            int(hidden_size / num_attention_heads)
        ]

        output_shapes = {output_name: last_state_shape}
        for i in range(num_layer):
            output_shapes["present_" + str(i)] = present_state_shape

        return output_shapes

    @staticmethod
    def auto_increase_buffer_size(output_buffers, output_shapes):
        for key in output_shapes:
            assert key in output_buffers
            buffer = output_buffers[key]
            if numpy.prod(output_shapes[key]) > buffer.nelement():
                output_buffers[key] = torch.empty(numpy.prod(output_shapes[key]),
                                                  dtype=buffer.dtype,
                                                  device=buffer.device)

    @staticmethod
    def get_output_buffers(output_shapes, device, is_float16=False):
        """ Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape.
        """
        data_type = torch.float16 if is_float16 else torch.float32

        output_buffers = {}
        for name, shape in output_shapes.items():
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=data_type, device=device)
        return output_buffers

    @staticmethod
    def diff_outputs(torch_outputs, ort_outputs, relative=False):
        """ Returns the maximum difference between PyTorch and OnnxRuntime outputs.
        """
        expected_outputs = torch_outputs[0].cpu().numpy()
        diff = numpy.abs(expected_outputs - ort_outputs[0])
        if relative:
            return numpy.amax(diff / (numpy.abs(expected_outputs) + 1e-6))
        else:
            return numpy.amax(diff)

    @staticmethod
    def compare_outputs(torch_outputs, ort_outputs, rtol=1e-03, atol=1e-03, has_beam_search=False):
        """ Returns True if torch and ORT outputs are close for given thresholds, and False otherwise.
        """
        is_close = numpy.allclose(ort_outputs[0], torch_outputs[0].cpu().numpy(), rtol=rtol, atol=atol)
        logger.debug(f'PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}')

        is_all_close = is_close
        if has_beam_search:
            num_layers = len(ort_outputs) - 4
        else:
            num_layers = len(ort_outputs) - 1
        for layer in range(num_layers):
            is_close = numpy.allclose(ort_outputs[1 + layer],
                                      torch_outputs[1][layer].cpu().numpy(),
                                      rtol=rtol,
                                      atol=atol)
            logger.debug(f'PyTorch and OnnxRuntime layer {layer} state (present_{layer}) are close:{is_close}')
            is_all_close = is_all_close and is_close

        if not is_all_close:
            max_abs_diff = Gpt2Helper.diff_outputs(torch_outputs, ort_outputs)
            logger.info(f'PyTorch and OnnxRuntime results are not all close: max_abs_diff={max_abs_diff:.5f}')

        return is_all_close

    @staticmethod
    def export_onnx(model,
                    device,
                    onnx_model_path: str,
                    verbose: bool = False,
                    use_external_data_format: bool = False,
                    has_position_ids: bool = True,
                    has_attention_mask: bool = True,
                    has_beam_select_idx: bool = True,
                    has_beam_search: bool = True):
        """ Export GPT-2 model with past state to ONNX model.
        """
        config: GPT2Config = model.config
        num_layer = config.n_layer
        dummy_inputs = Gpt2Helper.get_dummy_inputs(batch_size=1,
                                                   past_sequence_length=1,
                                                   sequence_length=1,
                                                   num_attention_heads=config.num_attention_heads,
                                                   hidden_size=config.hidden_size,
                                                   num_layer=num_layer,
                                                   vocab_size=config.vocab_size,
                                                   device=device,
                                                   float16=False,
                                                   has_position_ids=has_position_ids,
                                                   has_attention_mask=has_attention_mask,
                                                   has_beam_select_idx=has_beam_select_idx,
                                                   has_beam_search=has_beam_search)
        input_list = dummy_inputs.to_list()
        #input_ids, position_id, attention_mask, beam_select_idx, past = dummy_inputs.to_kwargs()
       
        with torch.no_grad():
            #outputs = model(input_ids, position_id, attention_mask, beam_select_idx, past)
            outputs = model(*input_list)

        past_names = [f'past_{i}' for i in range(num_layer)]
        present_names = [f'present_{i}' for i in range(num_layer)]

        # GPT2Model outputs last_state; GPT2LMHeadModel outputs logits (prediction_scores)
        if has_beam_search:
            output_names = ["logits"] + present_names
        else:
            assert outputs[0].shape[1] == config.vocab_size or outputs[0].shape[1] == config.hidden_size
            output_names = ["logits" if outputs[0].shape[1] == config.vocab_size else "last_state"] + present_names

        if has_beam_search:
            output_names += ["output_selected_indices", "output_log_probs", "output_unfinished_sents"]

        # Shape of input tensors:
        #    input_ids: (batch_size, seq_len)
        #    past_{i}:  (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
        #    attention_mask: (batch_size, past_seq_len + seq_len)
        # Shape of output tensors:
        #    last_state: (batch_size, seq_len, hidden_size)
        #      or logits: (batch_size, seq_len, vocab_size)
        #    present_{i}:  (2, batch_size, num_heads, past_seq_len + seq_len, hidden_size/num_heads)
        dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'seq_len'}, output_names[0]: {0: 'batch_size', 1: 'seq_len'}}
        for name in past_names:
            dynamic_axes[name] = {1: 'batch_size', 3: 'past_seq_len'}
        for name in present_names:
            dynamic_axes[name] = {1: 'batch_size', 3: 'total_seq_len'}

        input_names = ['input_ids']
        if has_position_ids:
            dynamic_axes['position_ids'] = {0: 'batch_size', 1: 'seq_len'}
            input_names.append('position_ids')
        if has_attention_mask:
            dynamic_axes['attention_mask'] = {0: 'batch_size', 1: 'total_seq_len'}
            input_names.append('attention_mask')
        if has_beam_select_idx:
            dynamic_axes['beam_select_idx'] = {1: 'batch_size'}
            input_names.append('beam_select_idx')            
        if has_beam_search:
            dynamic_axes['input_log_probs'] = {0: 'batch_size', 1: 'beam_size'}
            input_names.append('input_log_probs')   
            dynamic_axes['input_unfinished_sents'] = {0: 'batch_size', 1: 'beam_size'}
            input_names.append('input_unfinished_sents')  
        input_names.extend(past_names)

        logger.info(
            f"Shapes: input_ids={dummy_inputs.input_ids.shape} past={dummy_inputs.past[0].shape} output={outputs[0].shape} present={outputs[1][0].shape}"
        )

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(model,
                          args=tuple(input_list),
                          f=onnx_model_path,
                          input_names=input_names,
                          output_names=output_names,
                          example_outputs=outputs,
                          dynamic_axes=dynamic_axes,
                          opset_version=12,
                          do_constant_folding=True,
                          use_external_data_format=use_external_data_format,
                          verbose=verbose)
           
    @staticmethod
    def optimize_onnx(onnx_model_path,
                      optimized_model_path,
                      is_float16,
                      num_attention_heads,
                      hidden_size,
                      use_external_data_format=False):
        """ Optimize ONNX model with an option to convert it to use mixed precision.
        """
        from optimizer import optimize_model
        m = optimize_model(onnx_model_path,
                           model_type='gpt2',
                           num_heads=num_attention_heads,
                           hidden_size=hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=False)
        if is_float16:
            m.convert_model_float32_to_float16(cast_input_output=False)

        m.save_model_to_file(optimized_model_path, use_external_data_format)

    @staticmethod
    def pytorch_inference(model, inputs: Gpt2Inputs, total_runs: int = 0):
        """ Run inference of PyTorch model, and returns average latency in ms when total_runs > 0 besides outputs.
        """
        logger.debug("start pytorch_inference")

        # Convert it to fp32 as the PyTroch model cannot deal with half input.
        input_list = inputs.to_fp32().to_list()

        with torch.no_grad():
            outputs = model(*input_list)

        if total_runs == 0:
            return outputs

        latency = []
        with torch.no_grad():
            for _ in range(total_runs):
                start = time.time()
                outputs = model(*input_list)
                latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("PyTorch inference time = {} ms".format(format(average_latency, '.2f')))

        return outputs, average_latency

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: Gpt2Inputs, total_runs: int = 0):
        """ Run inference of ONNX model, and returns average latency in ms when total_runs > 0 besides outputs.
        """
        logger.debug(f"start onnxruntime_inference")

        ort_inputs = {'input_ids': numpy.ascontiguousarray(inputs.input_ids.cpu().numpy())}

        if inputs.past is not None:
            for i, past_i in enumerate(inputs.past):
                ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())

        if inputs.attention_mask is not None:
            ort_inputs['attention_mask'] = numpy.ascontiguousarray(inputs.attention_mask.cpu().numpy())

        if inputs.position_ids is not None:
            ort_inputs['position_ids'] = numpy.ascontiguousarray(inputs.position_ids.cpu().numpy())

        if inputs.beam_select_idx is not None:
            ort_inputs['beam_select_idx'] = numpy.ascontiguousarray(inputs.beam_select_idx.cpu().numpy())

        if inputs.input_log_probs is not None:
            ort_inputs['input_log_probs'] = numpy.ascontiguousarray(inputs.input_log_probs.cpu().numpy())

        if inputs.input_unfinished_sents is not None:
            ort_inputs['input_unfinished_sents'] = numpy.ascontiguousarray(inputs.input_unfinished_sents.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            ort_outputs = ort_session.run(None, ort_inputs)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime Inference time = {} ms".format(format(average_latency, '.2f')))

        return ort_outputs, average_latency

    @staticmethod
    def prepare_io_binding(
        ort_session, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past, 
        beam_select_idx, 
        input_log_probs, 
        input_unfinished_sents, 
        output_buffers, 
        output_shapes
    ):
        """ Returnas IO binding object for a session.
        """

        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_session.io_binding()

        if beam_select_idx is not None:
            assert beam_select_idx.is_contiguous()
            io_binding.bind_input('beam_select_idx', beam_select_idx.device.type, 0, numpy.longlong,
                                  list(beam_select_idx.size()), beam_select_idx.data_ptr())

        if input_log_probs is not None:
            assert input_log_probs.is_contiguous()
            io_binding.bind_input('input_log_probs', input_log_probs.device.type, 0, numpy.longlong,
                                  list(input_log_probs.size()), input_log_probs.data_ptr())

        if input_unfinished_sents is not None:
            assert input_unfinished_sents.is_contiguous()
            io_binding.bind_input('input_unfinished_sents', input_unfinished_sents.device.type, 0, numpy.longlong,
                                  list(input_unfinished_sents.size()), input_unfinished_sents.data_ptr())

        # Bind inputs
        assert input_ids.is_contiguous()
        io_binding.bind_input('input_ids', input_ids.device.type, 0, numpy.longlong, list(input_ids.size()),
                              input_ids.data_ptr())

        data_type = output_buffers[ort_session.get_outputs()[0].name].dtype
        float_type = numpy.float16 if data_type == torch.float16 else numpy.float32

        if past is not None:
            for i, past_i in enumerate(past):
                assert past_i.is_contiguous()
                io_binding.bind_input(f'past_{i}', past_i.device.type, 0, float_type, list(past_i.size()),
                                      past_i.data_ptr())

        if attention_mask is not None:
            assert attention_mask.is_contiguous()
            io_binding.bind_input('attention_mask', attention_mask.device.type, 0, float_type,
                                  list(attention_mask.size()), attention_mask.data_ptr())

        if position_ids is not None:
            assert position_ids.is_contiguous()
            io_binding.bind_input('position_ids', position_ids.device.type, 0, numpy.longlong,
                                  list(position_ids.size()), position_ids.data_ptr())

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(output_name, output_buffer.device.type, 0, float_type, output_shapes[output_name],
                                   output_buffer.data_ptr())

        return io_binding

    @staticmethod
    def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
        """ Copy results to cpu. Returns a list of numpy array.
        """
        ort_outputs = []
        for output in ort_session.get_outputs():
            output_name = output.name
            buffer = output_buffers[output_name]
            shape = output_shapes[output_name]
            copy_tensor = buffer[0:numpy.prod(shape)].reshape(shape).clone().detach()
            if return_numpy:
                ort_outputs.append(copy_tensor.cpu().numpy())
            else:
                ort_outputs.append(copy_tensor)
        return ort_outputs

    @staticmethod
    def onnxruntime_inference_with_binded_io(ort_session,
                                             inputs: Gpt2Inputs,
                                             output_buffers: Dict[str, torch.Tensor],
                                             output_shapes: Dict[str, List[int]],
                                             total_runs: int = 0,
                                             return_numpy: bool = True,
                                             include_copy_output_latency: bool = False):
        """ Inference with IO binding. Returns outputs, and optional latency when total_runs > 0.
        """
        logger.debug(f"start onnxruntime_inference_with_binded_io")

        # Bind inputs and outputs to onnxruntime session
        io_binding = Gpt2Helper.prepare_io_binding(
            ort_session, 
            inputs.input_ids, 
            inputs.position_ids,
            inputs.attention_mask, 
            inputs.past, 
            inputs.beam_select_idx, 
            inputs.input_log_probs, 
            inputs.input_unfinished_sents, 
            output_buffers, 
            output_shapes
        )

        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)

        # Copy results to cpu for verification
        ort_outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes,
                                                                    return_numpy)

        if total_runs == 0:
            return ort_outputs

        latency = []
        for _ in range(total_runs):
            start = time.time()
            # Run onnxruntime with io binding
            ort_session.run_with_iobinding(io_binding)
            if include_copy_output_latency:
                _ = Gpt2Helper.get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes,
                                                                  return_numpy)
            latency.append(time.time() - start)

        average_latency = sum(latency) * 1000 / len(latency)
        logger.debug("OnnxRuntime with IO binding inference time = {} ms".format(format(average_latency, '.2f')))

        return ort_outputs, average_latency

    @staticmethod
    def test_parity(ort_session,
                    model,
                    device,
                    is_float16=False,
                    rtol=5e-4,
                    atol=5e-4,
                    total_test_cases=100,
                    use_io_binding=True,
                    model_class="GPT2LMHeadModel",
                    has_position_ids=True,
                    has_attention_mask=True,
                    has_beam_select_idx=True,
                    has_beam_search=True):
        """ Generate random inputs and compare the results of PyTorch and Onnx Runtime.
        """

        config: GPT2Config = model.config

        logger.info(
            f"Running parity test (rtol={rtol}, atol={atol}, test_cases={total_test_cases}, use_io_binding={use_io_binding} model_class={model_class} is_float16={is_float16}) ..."
        )

        max_batch_size = 8
        max_past_seq_len = 4  # Do not use large number here for higher chance of hitting empty past (past_seq_len=0)
        max_seq_len = 2

        output_buffers = None
        if use_io_binding:
            max_output_shapes = Gpt2Helper.get_output_shapes(max_batch_size, max_past_seq_len, max_seq_len, config,
                                                             model_class)
            output_buffers = Gpt2Helper.get_output_buffers(max_output_shapes, device, is_float16)

        passed_test_cases = 0
        for _ in range(total_test_cases):
            sequence_length = random.randint(1, max_seq_len)
            past_sequence_length = random.randint(0, max_past_seq_len)
            batch_size = random.randint(1, max_batch_size)

            logger.debug(
                f"Running parity test for batch_size={batch_size} past_sequence_length={past_sequence_length}...")
            dummy_inputs = Gpt2Helper.get_dummy_inputs(batch_size, past_sequence_length, sequence_length,
                                                       config.num_attention_heads, config.hidden_size, config.n_layer,
                                                       config.vocab_size, device, is_float16, has_position_ids,
                                                       has_attention_mask, has_beam_select_idx, has_beam_search)

            outputs = Gpt2Helper.pytorch_inference(model, dummy_inputs)
            if use_io_binding:
                ort_outputs = Gpt2Helper.onnxruntime_inference(ort_session, dummy_inputs)
            else:
                output_shapes = Gpt2Helper.get_output_shapes(batch_size, past_sequence_length, sequence_length, config,
                                                             model_class)
                ort_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(ort_session, dummy_inputs, output_buffers,
                                                                              output_shapes)

            is_all_close = Gpt2Helper.compare_outputs(outputs, ort_outputs, rtol=rtol, atol=atol, has_beam_search=has_beam_search)
            if is_all_close:
                passed_test_cases += 1
        logger.info(f"Parity Test Cases={total_test_cases}; Passed={passed_test_cases}")
        if passed_test_cases > 0.95 * total_test_cases:
            logger.info(f"Parity is good: passed rate={int(passed_test_cases*100/total_test_cases):.0f}%")
        return passed_test_cases == total_test_cases

    @staticmethod
    def torchscript(model, config, device, has_position_ids=True, has_attention_mask=True):
        """ JIT trace for TorchScript.
        """
        input_list = Gpt2Helper.get_dummy_inputs(batch_size=1,
                                                 past_sequence_length=1,
                                                 sequence_length=1,
                                                 num_attention_heads=config.num_attention_heads,
                                                 hidden_size=config.hidden_size,
                                                 num_layer=config.n_layer,
                                                 vocab_size=config.vocab_size,
                                                 device=device,
                                                 float16=False,
                                                 has_position_ids=has_position_ids,
                                                 has_attention_mask=has_attention_mask).to_list()
        return torch.jit.trace(model, input_list)

    @staticmethod
    def get_onnx_paths(output_dir,
                       model_name_or_path,
                       model_class: str = 'GPT2LMHeadModel',
                       has_past=True,
                       new_folder=False):
        """ Build a  path name for given model based on given attributes.
        """
        model_name = model_name_or_path
        if not re.match('^[\w_-]+$', model_name_or_path):  # It is not a name, shall be a path
            assert os.path.isdir(model_name_or_path)
            model_name = Path(model_name_or_path).parts[-1]

        if model_class != 'GPT2LMHeadModel':
            model_name += "_" + model_class

        if has_past:
            model_name += "_past"

        if new_folder:
            # store each model to its own directory (for external data format).
            return {
                "raw": os.path.join(os.path.join(output_dir, model_name), model_name + ".onnx"),
                "fp32": os.path.join(os.path.join(output_dir, model_name + "_fp32"), model_name + "_fp32.onnx"),
                "fp16": os.path.join(os.path.join(output_dir, model_name + "_fp16"), model_name + "_fp16.onnx"),
                "int8": os.path.join(os.path.join(output_dir, model_name + "_int8"), model_name + "_int8.onnx")
            }

        return {
            "raw": os.path.join(output_dir, model_name + ".onnx"),
            "fp32": os.path.join(output_dir, model_name + "_fp32.onnx"),
            "fp16": os.path.join(output_dir, model_name + "_fp16.onnx"),
            "int8": os.path.join(output_dir, model_name + "_int8.onnx")
        }
