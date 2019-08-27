# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# Copyright (c) 2019, Guanlong Zhao
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Modified from https://github.com/NVIDIA/tacotron2"""

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from common.layers import ConvNorm, LinearNorm
from common.utils import to_gpu, get_mask_from_lengths,\
    get_mask_from_lengths_window_and_time_step
from common.fp16_optimizer import fp32_to_fp16, fp16_to_fp32


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        self.not_so_small_mask = -1000

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_acoustic_feat_dims,
                         hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim,
                         hparams.n_acoustic_feat_dims,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_acoustic_feat_dims))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5,
                          self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.prenet = Prenet(hparams.n_symbols,
                             [hparams.symbols_embedding_dim,
                              hparams.symbols_embedding_dim])

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        # x: (B, D, T) -> (B, T, D) -> (B, D, T)
        x = self.prenet(x.transpose(1, 2)).transpose(1, 2)

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        # x: (B, D, T) -> (B, T, D) -> (B, D, T)
        x = self.prenet(x.transpose(1, 2)).transpose(1, 2)

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_acoustic_feat_dims = hparams.n_acoustic_feat_dims
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.attention_window_size = hparams.attention_window_size
        self.prenet = Prenet(hparams.n_acoustic_feat_dims,
                             [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        # self.decoder_rnn2 = nn.LSTMCell(
        #     hparams.decoder_rnn_dim,
        #     hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_acoustic_feat_dims)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_acoustic_feat_dims).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. acoustic outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training,
        i.e. acoustic-feats.

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_acoustic_feat_dims, T_out) -> (B, T_out, n_acoustic_feat_dims)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)), -1)
        # (B, T_out, n_acoustic_feat_dims) -> (T_out, B, n_acoustic_feat_dims)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, acoustic_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        acoustic_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        acoustic_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_acoustic_feat_dims) -> (B, T_out, n_acoustic_feat_dims)
        acoustic_outputs = torch.stack(acoustic_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        acoustic_outputs = acoustic_outputs.view(
            acoustic_outputs.size(0), -1, self.n_acoustic_feat_dims)
        # (B, T_out, n_acoustic_feat_dims) -> (B, n_acoustic_feat_dims, T_out)
        acoustic_outputs = acoustic_outputs.transpose(1, 2)

        return acoustic_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_windowed_mask=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous acoustic output
        attention_window: B*T mask.

        RETURNS
        -------
        decoder_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        if attention_windowed_mask is None:
            self.attention_context, self.attention_weights = \
                self.attention_layer(self.attention_hidden, self.memory,
                                     self.processed_memory,
                                     attention_weights_cat, self.mask)
        else:
            self.attention_context, self.attention_weights = \
                self.attention_layer(self.attention_hidden, self.memory,
                                     self.processed_memory,
                                     attention_weights_cat,
                                     attention_windowed_mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        # self.decoder_hidden, self.decoder_cell = self.decoder_rnn2(
        #     self.decoder_hidden, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. acoustic-feats.
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        acoustic_outputs: acoustic outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        acoustic_outputs, gate_outputs, alignments = [], [], []
        while len(acoustic_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(acoustic_outputs)]

            if self.attention_window_size is not None:
                time_step = len(acoustic_outputs)
                attention_windowed_mask = \
                    get_mask_from_lengths_window_and_time_step(
                        memory_lengths, self.attention_window_size, time_step)
            else:
                attention_windowed_mask = None
            acoustic_output, gate_output, attention_weights = self.decode(
                decoder_input, attention_windowed_mask)
            acoustic_outputs += [acoustic_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        acoustic_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            acoustic_outputs, gate_outputs, alignments)

        return acoustic_outputs, gate_outputs, alignments

    def inference(self, memory, memory_lengths):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        acoustic_outputs: acoustic outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        acoustic_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)

            if self.attention_window_size is not None:
                time_step = len(acoustic_outputs)
                attention_windowed_mask = \
                    get_mask_from_lengths_window_and_time_step(
                        memory_lengths, self.attention_window_size, time_step)
            else:
                attention_windowed_mask = None

            acoustic_output, gate_output, alignment = self.decode(
                decoder_input, attention_windowed_mask)

            acoustic_outputs += [acoustic_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(acoustic_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = acoustic_output

        acoustic_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            acoustic_outputs, gate_outputs, alignments)

        return acoustic_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_acoustic_feat_dims = hparams.n_acoustic_feat_dims
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        ppg_padded, input_lengths, acoustic_padded, gate_padded,\
            output_lengths = batch
        ppg_padded = to_gpu(ppg_padded).float()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        acoustic_padded = to_gpu(acoustic_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return ((ppg_padded, input_lengths, acoustic_padded, max_len,
                 output_lengths),
                (acoustic_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_acoustic_feat_dims, mask.size(0),
                               mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, \
            output_lengths = self.parse_input(inputs)
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        # inputs: (B, D, T)
        encoder_outputs = self.encoder(inputs, input_lengths)

        acoustic_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)

        acoustic_outputs_postnet = self.postnet(acoustic_outputs)
        acoustic_outputs_postnet = acoustic_outputs + acoustic_outputs_postnet

        return self.parse_output([acoustic_outputs, acoustic_outputs_postnet,
                                  gate_outputs, alignments], output_lengths)

    def inference(self, inputs):
        inputs = self.parse_input(inputs)
        input_lengths = torch.cuda.LongTensor([t.shape[1] for t in inputs])
        encoder_outputs = self.encoder.inference(inputs)
        acoustic_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, input_lengths)

        acoustic_outputs_postnet = self.postnet(acoustic_outputs)
        acoustic_outputs_postnet = acoustic_outputs + acoustic_outputs_postnet

        outputs = self.parse_output([acoustic_outputs, acoustic_outputs_postnet,
                                     gate_outputs, alignments])

        return outputs


class DecoderPPG2PPG(nn.Module):
    def __init__(self, hparams):
        super(DecoderPPG2PPG, self).__init__()
        self.n_acoustic_feat_dims = hparams.n_acoustic_feat_dims
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.attention_window_size = hparams.attention_window_size
        self.prenet = Prenet(hparams.n_acoustic_feat_dims,
                             [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = Prenet(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            [hparams.symbols_embedding_dim, hparams.n_acoustic_feat_dims])

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_acoustic_feat_dims).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. acoustic outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training,
        i.e. acoustic-feats.

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_acoustic_feat_dims, T_out) -> (B, T_out, n_acoustic_feat_dims)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)), -1)
        # (B, T_out, n_acoustic_feat_dims) -> (T_out, B, n_acoustic_feat_dims)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, acoustic_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        acoustic_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        acoustic_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_acoustic_feat_dims) -> (B, T_out, n_acoustic_feat_dims)
        acoustic_outputs = torch.stack(acoustic_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        acoustic_outputs = acoustic_outputs.view(
            acoustic_outputs.size(0), -1, self.n_acoustic_feat_dims)
        # (B, T_out, n_acoustic_feat_dims) -> (B, n_acoustic_feat_dims, T_out)
        acoustic_outputs = acoustic_outputs.transpose(1, 2)

        return acoustic_outputs, gate_outputs, alignments

    def decode(self, decoder_input, attention_windowed_mask=None):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous acoustic output
        attention_window: B*T mask.

        RETURNS
        -------
        decoder_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)

        if attention_windowed_mask is None:
            self.attention_context, self.attention_weights = \
                self.attention_layer(self.attention_hidden, self.memory,
                                     self.processed_memory,
                                     attention_weights_cat, self.mask)
        else:
            self.attention_context, self.attention_weights = \
                self.attention_layer(self.attention_hidden, self.memory,
                                     self.processed_memory,
                                     attention_weights_cat,
                                     attention_windowed_mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. acoustic-feats.
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        acoustic_outputs: acoustic outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        acoustic_outputs, gate_outputs, alignments = [], [], []
        while len(acoustic_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(acoustic_outputs)]

            if self.attention_window_size is not None:
                time_step = len(acoustic_outputs)
                attention_windowed_mask = \
                    get_mask_from_lengths_window_and_time_step(
                        memory_lengths, self.attention_window_size, time_step)
            else:
                attention_windowed_mask = None
            acoustic_output, gate_output, attention_weights = self.decode(
                decoder_input, attention_windowed_mask)
            acoustic_outputs += [acoustic_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        acoustic_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            acoustic_outputs, gate_outputs, alignments)

        return acoustic_outputs, gate_outputs, alignments

    def inference(self, memory, memory_lengths):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        acoustic_outputs: acoustic outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        acoustic_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)

            if self.attention_window_size is not None:
                time_step = len(acoustic_outputs)
                attention_windowed_mask = \
                    get_mask_from_lengths_window_and_time_step(
                        memory_lengths, self.attention_window_size, time_step)
            else:
                attention_windowed_mask = None

            acoustic_output, gate_output, alignment = self.decode(
                decoder_input, attention_windowed_mask)

            acoustic_outputs += [acoustic_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(acoustic_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = acoustic_output

        acoustic_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            acoustic_outputs, gate_outputs, alignments)

        return acoustic_outputs, gate_outputs, alignments


class PPG2PPG(nn.Module):
    def __init__(self, hparams):
        super(PPG2PPG, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_acoustic_feat_dims = hparams.n_acoustic_feat_dims
        self.encoder = Encoder(hparams)
        self.decoder = DecoderPPG2PPG(hparams)

    def parse_batch(self, batch):
        ppg_padded, input_lengths, acoustic_padded, gate_padded,\
            output_lengths = batch
        ppg_padded = to_gpu(ppg_padded).float()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        acoustic_padded = to_gpu(acoustic_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return ((ppg_padded, input_lengths, acoustic_padded, max_len,
                 output_lengths),
                (acoustic_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_acoustic_feat_dims, mask.size(0),
                               mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, \
            output_lengths = self.parse_input(inputs)
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        # inputs: (B, D, T)
        encoder_outputs = self.encoder(inputs, input_lengths)

        acoustic_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)

        # (B, D, T)
        # acoustic_outputs = F.log_softmax(acoustic_outputs, dim=1)
        acoustic_outputs = F.softmax(acoustic_outputs, dim=1)

        return self.parse_output([acoustic_outputs, gate_outputs,
                                  alignments], output_lengths)

    def inference(self, inputs):
        inputs = self.parse_input(inputs)
        input_lengths = torch.cuda.LongTensor([t.shape[1] for t in inputs])
        encoder_outputs = self.encoder.inference(inputs)
        acoustic_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, input_lengths)

        # (B, D, T)
        # acoustic_outputs = F.log_softmax(acoustic_outputs, dim=1)
        acoustic_outputs = F.softmax(acoustic_outputs, dim=1)

        outputs = self.parse_output([acoustic_outputs, gate_outputs,
                                     alignments])

        return outputs
