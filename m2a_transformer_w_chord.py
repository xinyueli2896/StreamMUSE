import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.roformer.modeling_roformer import RoFormerModel, RoFormerConfig, RoFormerEncoder
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import sys
import pdb
import wandb
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
# from preprocess_large_midi_dataset import tensor_to_midi
import argparse
import time
from preprocess_large_midi_dataset import DURATION_TEMPLATES
from rwc_preprocessing import *
from generate_chroma import *
def decode_output(outputs, save_path, tempo=120.0, prompt=True, single=False):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    time_step_length = 60.0 / tempo / 4
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    for output in outputs:
        instrument_map = {}
        for time_step, data in enumerate(output):
            content = data.squeeze(0)
            time_step = time_step if single else time_step//2
            start_time = time_step * time_step_length
            for i in range(0, len(content), 2):
                program = int(content[i].item())
                if program ==EOS_TOKEN:
                    break
                if i + 1 >= len(content):
                    print('Incomplete note @', time_step, i)
                    break
                pitch_duration = int(content[i+1].item())
                pitch_duration = pitch_duration - 2
                pitch = pitch_duration % 128
                duration = pitch_duration // 128
                if program != 0 and program != 1:
                    print('Invalid program:', program, '@', time_step, i)
                    break
                if pitch < 0 or pitch >= 128:
                    print('Invalid pitch:', pitch, '@', time_step, i)
                    break
                if duration < 0 or duration >= len(DURATION_TEMPLATES):
                    print('Invalid duration:', duration, '@', time_step, i)
                    break
                end_time = DURATION_TEMPLATES[duration] * time_step_length + start_time
                if program not in instrument_map:
                    if program == 0:
                        inst = pretty_midi.Instrument(program=24, name='Guitar')
                    else:  # program == 1
                        inst = pretty_midi.Instrument(program=0, name='Piano')
                    instrument_map[program] = inst
                    midi.instruments.append(inst)

                inst = instrument_map[program]
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    midi.write(save_path)
TRAIN_LENGTH = 192
MAX_STEPS = 1000000

# Indicator: 0
# pitch+duration*2: 3200 (25*128)
N_NORMAL_TOKENS = 3202
N_TOKENS = N_NORMAL_TOKENS + 3
SOS_TOKEN = N_NORMAL_TOKENS
EOS_TOKEN = N_NORMAL_TOKENS + 1
PAD_TOKEN = N_NORMAL_TOKENS + 2

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)

class RoFormerSymbolicTransformer(L.LightningModule):

    def __init__(self, large=False):
        super().__init__()
        self.hidden_size= 768 if large else 512
        self.chord_hidden_size = 64
        self.num_layers = 12 if large else 6
        self.num_attention_heads = 12 if large else 8
        self.intermediate_size = 3072 if large else 1024
        self.local_model_num_layers = 3
        self.local_model_num_attention_heads = 8
        self.local_model_intermediate_size = 768
        main_roformer_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.model = self.get_base_model(main_roformer_config)
        local_encoder_config = local_decoder_config = RoFormerConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.local_model_num_layers,
            num_attention_heads=self.local_model_num_attention_heads,
            intermediate_size=self.local_model_intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.local_embedding = nn.Embedding(N_TOKENS, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, self.hidden_size)
        with torch.no_grad():
            self.token_type_embeddings.weight.mul_(2.0)
        # self.token_type_embeddings.weight.requires_grad_(True)
        self.local_encoder = RoFormerEncoder(local_encoder_config)
        self.local_decoder = RoFormerEncoder(local_decoder_config)
        self.final_decoder = nn.Linear(self.hidden_size, N_TOKENS)
        self.global_sos = nn.Parameter(torch.randn(self.hidden_size))
        self._future_mask = torch.empty(0)
        self.chord_proj = nn.Linear(12, self.chord_hidden_size)
        self.fuse = nn.Linear(self.hidden_size+self.chord_hidden_size, self.hidden_size)
        # self.type_classifier = nn.Linear(self.hidden_size, 2)
        # self.type_classifier.weight.requires_grad_(False)

        # self.type_scale = nn.Parameter(torch.tensor(1.0))
        # self.mix_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # self.type_proj = nn.Sequential(
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.ReLU()
        # )

    def get_base_model(self, config):
        return RoFormerEncoder(config)

    def local_encode(self, x, token_type_ids):

        batch_size, seq_len, subseq_len = x.shape
        x = x.view(-1, subseq_len)

        # prepend SOS:
        x = torch.cat([
            torch.full((x.shape[0], 1), SOS_TOKEN, dtype=torch.long, device=x.device),
            x
        ], dim=-1)  # now [B*seq_len, subseq_len+1]

        mask = x != PAD_TOKEN     # [B*seq_len, subseq_len+1]
        word_emb = self.local_embedding(x)  # → [B*seq_len, subseq_len+1, H]

        type_emb = self.token_type_embeddings(token_type_ids)      # [B*seq_len, subseq_len+1, H]
        type_emb = type_emb.view(batch_size*seq_len, word_emb.shape[1], -1)

        emb = word_emb + type_emb
        # print("x_chord", x_chord.type())
        # print("x", x.type())

        # x_chord = x_chord.to(self.chord_proj.weight.dtype)
        # chord_h = self.chord_proj(x_chord)      # [B, seq_len, H]
        # # chord_h = chord_h.view(batch_size*seq_len, 1, -1)  # [B*seq_len,1,H]
        # # chord_h = chord_h.expand(-1, emb.size(1), -1)      # [B*seq_len, L, H]
        # print("before concat", emb.shape)
        # # concat along feature dim:
        # emb = torch.cat([emb, chord_h], dim=-1)  # [B*seq_len, L, 2H]
        # print("after concat", emb.shape)
        # emb = self.fuse(emb)  # → [B*seq_len, L, H]

        h = self.local_encoder(emb, encoder_attention_mask=mask)[0]

        return h[:, 0], emb[:, :-1]

    def local_decode(self, h, emb):
        batch_size, subseq_len, _ = emb.shape
        # Add h as the first token of emb
        h = h.view(batch_size, 1, -1)
        # print(emb.shape) #3840 8 512
        emb = torch.cat([h, emb[:, 1:]], dim=1)
        # Create an autoregressive mask
        h = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
        return self.final_decoder(h)


    def local_sampling(self, h, max_subseq_len=32, temperature=1.0):
        batch_size, _ = h.shape
        y = torch.zeros((batch_size, 0), dtype=torch.long, device=h.device)
        emb = h[:, None, :]
        eos_triggered = torch.zeros(batch_size, dtype=torch.bool, device=h.device)

        for i in range(max_subseq_len):
            h_ = self.local_decoder(emb, attention_mask=self.buffered_future_mask(emb))[0]
            if temperature == 0:
                p = F.one_hot(self.final_decoder(h_).argmax(dim=-1), N_TOKENS).float()
            else:
                p = F.softmax(self.final_decoder(h_[:, -1]) / temperature, dim=-1)
            y_next = torch.multinomial(p, 1)
            y_next[eos_triggered, :] = PAD_TOKEN
            eos_triggered = eos_triggered | (y_next.squeeze(1) == EOS_TOKEN)
            y = torch.cat([y, y_next], dim=1)
            if torch.all(eos_triggered):
                break

            # 5a) now append the embedding (always ACCOMPANIMENT), so token_type_ids = 1
            emb = torch.cat([emb, self.local_embedding(y_next) + self.token_type_embeddings(torch.ones_like(y_next))], dim=1)

        return y
   

    def global_sampling(self, x, x_chord=None, x_mel_gt=None, max_seq_len=384, temperature=1.0, prompt_len=75):
        
        batch_size, seq_len, subseq_len = x.shape
        print(x.shape) #prompt*2
        # for i in range(seq_len):
        #     if i%2==0:
        #         x[:, i] = torch.tensor([3203, 3204, 3202, 3204 , 3204, 3204, 3204, 3204]).cuda()
        #         x_chord[:, i] = torch.zeros((12)).cuda()
        _, seq_len_gt, _ = x_mel_gt.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, _= self.local_encode(x, token_type_ids)
        h_mel, _ = self.local_encode(x_mel_gt, torch.zeros(*x_mel_gt.shape[:-1], x_mel_gt.shape[-1] + 1, device=x_mel_gt.device, dtype=x_mel_gt.dtype))
        h = h.view(batch_size, seq_len, -1)
        h_mel = h_mel.view(batch_size, seq_len_gt, -1)
        # print("1",x_chord.shape)
        x_chord = x_chord.to(self.chord_proj.weight.dtype)
        x_chord = self.chord_proj(x_chord)
        # print("2",x_chord.shape)
        h = torch.cat([h, x_chord[:, :prompt_len*2]], dim=-1)
        h = self.fuse(h)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2
        if x_mel_gt != None:
            print('with gt!')
            frame_times = []
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    print('Sampling', i, '/', max_seq_len)
                if i % 2 == 0:
                    if h.device.type == 'cuda': torch.cuda.synchronize()
                    t0 = time.perf_counter()
            
                    h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                    y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                    y.append(y_next)
                    b, s, l = y_next.unsqueeze(1).shape
                    token_type_ids = torch.ones((b, s, l+1), dtype=torch.long, device=y_next.device)
                    next_h = self.local_encode(y_next.unsqueeze(1), token_type_ids = token_type_ids)[0].unsqueeze(1)
                    # print("3", next_h.shape)
                    # print("4", x_chord[:, prompt_len*2+i].shape)
                    next_h = self.fuse(torch.cat([next_h, x_chord[:, prompt_len*2+i].unsqueeze(1)], dim=-1))
                    h = torch.cat([h, next_h], dim=1)
                    if h.device.type == 'cuda': torch.cuda.synchronize()
                    frame_times.append(time.perf_counter() - t0)
                else:
                    # token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                    h_prev_mel = h_mel[:, i//2, :].unsqueeze(1)  # [B, 1, H]
                    h_prev_mel = self.fuse(torch.cat([h_prev_mel, x_chord[:, prompt_len*2+i].unsqueeze(1)], dim=-1))
                    h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                    y.append(x_mel_gt[: ,i//2, :])
        else:
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    print('Sampling', i, '/', max_seq_len)
                h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                y.append(y_next)
                b, s, l = y_next.unsqueeze(1).shape
                if i%2==0:
                    token_type_ids = torch.ones((b, s, l+1), dtype=torch.long, device=y_next.device)
                else:
                    token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids = token_type_ids)[0].unsqueeze(1)], dim=1)
        times = torch.tensor(frame_times)
        avg = times.mean().item() * 1000   # ms
        p50 = times.median().item() * 1000
        print(f"Avg per-frame: {avg:.2f} ms, 50%-ile: {p50:.2f} ms over {len(times)} frames")
        return y
    def global_sampling_with_mel_only(self, x, x_chord=None, x_mel_gt=None, max_seq_len=384, temperature=1.0, prompt_len=75):
        
        batch_size, seq_len, subseq_len = x.shape
        print(x.shape)
        # for i in range(seq_len):
        #     if i%2==0:
        #         x[:, i] = torch.tensor([3203, 3204, 3204, 3204 , 3204, 3204, 3204, 3204]).cuda()
        #         x_chord[:, i] = torch.zeros((12)).cuda()
        _, seq_len_gt, _ = x_mel_gt.shape
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, _= self.local_encode(x, token_type_ids)
        h_mel, _ = self.local_encode(x_mel_gt, torch.zeros(*x_mel_gt.shape[:-1], x_mel_gt.shape[-1] + 1, device=x_mel_gt.device, dtype=x_mel_gt.dtype))
        h = h.view(batch_size, seq_len, -1)
        h_mel = h_mel.view(batch_size, seq_len_gt, -1)
        # print("1",x_chord.shape)
        x_chord = x_chord.to(self.chord_proj.weight.dtype)
        x_chord = self.chord_proj(x_chord)
        # print("2",x_chord.shape)
        h = torch.cat([h, x_chord[:, :prompt_len*2]], dim=-1)
        h = self.fuse(h)
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h], dim=1)
        y = [x[:, i, :] for i in range(seq_len)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2
        if x_mel_gt != None:
            print('with gt!')
            frame_times = []
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    print('Sampling', i, '/', max_seq_len)
                if i % 2 == 0:
                    if h.device.type == 'cuda': torch.cuda.synchronize()
                    t0 = time.perf_counter()
            
                    h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                    y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                    y.append(y_next)
                    b, s, l = y_next.unsqueeze(1).shape
                    token_type_ids = torch.ones((b, s, l+1), dtype=torch.long, device=y_next.device)
                    next_h = self.local_encode(y_next.unsqueeze(1), token_type_ids = token_type_ids)[0].unsqueeze(1)
                    # print("3", next_h.shape)
                    # print("4", x_chord[:, prompt_len*2+i].shape)
                    next_h = self.fuse(torch.cat([next_h, x_chord[:, prompt_len*2+i].unsqueeze(1)], dim=-1))
                    h = torch.cat([h, next_h], dim=1)
                    if h.device.type == 'cuda': torch.cuda.synchronize()
                    frame_times.append(time.perf_counter() - t0)
                else:
                    # token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                    h_prev_mel = h_mel[:, i//2, :].unsqueeze(1)  # [B, 1, H]
                    h_prev_mel = self.fuse(torch.cat([h_prev_mel, x_chord[:, prompt_len*2+i].unsqueeze(1)], dim=-1))
                    h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                    y.append(x_mel_gt[: ,i//2, :])
        else:
            for i in range(0, max_seq_len):
                if i % 10 == 0:
                    print('Sampling', i, '/', max_seq_len)
                h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
                y_next = self.local_sampling(h_out[:, -1], max_subseq_len=subseq_len, temperature=temperature)
                y.append(y_next)
                b, s, l = y_next.unsqueeze(1).shape
                if i%2==0:
                    token_type_ids = torch.ones((b, s, l+1), dtype=torch.long, device=y_next.device)
                else:
                    token_type_ids = torch.zeros((b, s, l+1), dtype=torch.long, device=y_next.device)
                h = torch.cat([h, self.local_encode(y_next.unsqueeze(1), token_type_ids = token_type_ids)[0].unsqueeze(1)], dim=1)
        times = torch.tensor(frame_times)
        avg = times.mean().item() * 1000   # ms
        p50 = times.median().item() * 1000
        print(f"Avg per-frame: {avg:.2f} ms, 50%-ile: {p50:.2f} ms over {len(times)} frames")
        return y
    # def global_sampling_with_mel_only(self, x_mel: torch.LongTensor, x_chord: torch.LongTensor, temperature: float = 1.0, max_seq_len=384, prompt_len=16):
    #     B, S, L = x_mel.shape
    #     device = x_mel.device

    #     # Build program IDs = 0 for all melody tokens
    #     # token_type_ids = torch.zeros_like(x_mel, dtype=torch.long)  # [B, S, L]
    #     x_acc_pad = torch.tensor([3203, 3204, 3204, 3204 , 3204, 3204, 3204, 3204])
    #     x_acc_pad = x_acc_pad.repeat(B, prompt_len, 1).cuda()
    #     h_acc_pad, _ = self.local_encode(x_acc_pad, torch.ones((B, prompt_len, L+1), device=device, dtype=x_acc_pad.dtype))
    #     h_acc_pad = h_acc_pad.view(B, prompt_len, self.hidden_size)
    #     h_mel, _ = self.local_encode(x_mel, torch.zeros((B, S, L+1), device=device, dtype=x_mel.dtype))
    #     h_mel = h_mel.view(B, S, self.hidden_size)     # [B, S, H]
    #     x_chord = x_chord.to(self.chord_proj.weight.dtype)

    #     for i in range(prompt_len):
    #         x_chord[:, 2*i] = torch.zeros((12))
    #     x_chord = self.chord_proj(x_chord)
        
    #     h = torch.stack([h_acc_pad, h_mel[:, :prompt_len]], dim=2)
    #     h = h.view(B, prompt_len*2, self.hidden_size)
        
    #     h = self.fuse(torch.cat([h, x_chord[:, :prompt_len*2]], dim=-1))
    #     # Prepare SOS for global
    #     sos = self.global_sos.view(1, 1, -1).repeat(B, 1, 1)  # [B, 1, H]
    #     x = torch.stack([x_acc_pad, x_mel[:, :prompt_len]], dim=2).view(B, prompt_len*2, L)
    #     # Will store generated accompaniment frames
    #     y = [x[:, i, :] for i in range(prompt_len*2)]  # y will be returned by a list a0,m0,a1,m1,a_to_be_2 # each entry: [B, L]

    #     # Start with just [SOS]
    #     h = torch.cat([sos, h], dim=1)
    #     for t in range(max_seq_len):
    #         if t % 10 == 0:
    #                 print('Sampling', t, '/', max_seq_len)

    #         if t > 0 :
    #             # Append previous melody summary before generating new accompaniment
    #             h_prev_mel = h_mel[:, t-1, :].unsqueeze(1)  # [B, 1, H]
    #             h_prev_mel = self.fuse(torch.cat([h_prev_mel, x_chord[:, prompt_len+t-1].unsqueeze(1)], dim=-1))
    #             h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
    #             y.append(x_mel[: ,prompt_len+t-1, :])

    #         h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
    #         y_next = self.local_sampling(h_out[:, -1], max_subseq_len=L, temperature=temperature)
    #         y.append(y_next)
    #         b, s, l = y_next.unsqueeze(1).shape
    #         token_type_ids = torch.ones((b, s, l+1), dtype=torch.long, device=y_next.device)
    #         next_h = self.local_encode(y_next.unsqueeze(1), token_type_ids = token_type_ids)[0].unsqueeze(1)
    #         next_h = self.fuse(torch.cat([next_h, x_chord[:, prompt_len+t].unsqueeze(1)], dim=-1))
    #         h = torch.cat([h, next_h], dim=1)
    #     return y  # list of S tensors [B, L]

    def global_sampling_from_scratch(self, x_mel: torch.LongTensor, x_chord: torch.LongTensor, temperature: float = 1.0, max_seq_len=384):
        B, S, L = x_mel.shape
        device = x_mel.device

        # Build program IDs = 0 for all melody tokens
        # token_type_ids = torch.zeros_like(x_mel, dtype=torch.long)  # [B, S, L]
        h_mel, _ = self.local_encode(x_mel, torch.zeros((B, S, L+1), device=device, dtype=x_mel.dtype))
        h_mel = h_mel.view(B, S, self.hidden_size)     # [B, S, H]
        print(x_chord.shape)
        x_chord = x_chord.to(self.chord_proj.weight.dtype)
        x_chord = self.chord_proj(x_chord)
        # Prepare SOS for global
        sos = self.global_sos.view(1, 1, -1).repeat(B, 1, 1)  # [B, 1, H]

        # Will store generated accompaniment frames
        y = []  # each entry: [B, L]

        # Start with just [SOS]
        h = sos  # [B, 1, H]
        for t in range(max_seq_len):
            if t % 10 == 0:
                    print('Sampling', t, '/', max_seq_len)

            if t > 0 :
                # Append previous melody summary before generating new accompaniment
                h_prev_mel = h_mel[:, t-1, :].unsqueeze(1)  # [B, 1, H]
                h_prev_mel = self.fuse(torch.cat([h_prev_mel, x_chord[:, t-1].unsqueeze(1)], dim=-1))
                h = torch.cat([h, h_prev_mel], dim=1)  # [B, cur_len, H]
                y.append(x_mel[: ,t-1, :])

            h_out = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0]
            y_next = self.local_sampling(h_out[:, -1], max_subseq_len=L, temperature=temperature)
            y.append(y_next)
            b, s, l = y_next.unsqueeze(1).shape
            token_type_ids = torch.ones((b, s, l+1), dtype=torch.long, device=y_next.device)
            next_h = self.local_encode(y_next.unsqueeze(1), token_type_ids = token_type_ids)[0].unsqueeze(1)
            next_h = self.fuse(torch.cat([next_h, x_chord[:, t].unsqueeze(1)], dim=-1))
            h = torch.cat([h, next_h], dim=1)
        return y  # list of S tensors [B, L]


    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, x, x_chord):
        # x: [batch, seq, subseq]
        # Use local encoder to encode subsequences
        batch_size, seq_len, subseq_len = x.shape # 10*384*8
        assert seq_len % 2 == 0, "Expected even number of frames (2*S interleaved)."
        # print("x_chord1", x_chord.shape)
        idx = torch.arange(seq_len, device=x.device)
        frame_type = (idx % 2 == 0).long()  # → [seq_len], 1 at even idx (acc), 0 at odd idx (mel)
        token_type_ids = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, subseq_len)
        sos_type = frame_type.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        token_type_ids = torch.cat([sos_type, token_type_ids], dim=-1)
        h, emb= self.local_encode(x, token_type_ids) 
        h = h.view(batch_size, seq_len, -1) #这是每一帧的summary
        x_chord = x_chord.to(self.chord_proj.weight.dtype)
        # print("x_chord2", x_chord.shape)
        x_chord = self.chord_proj(x_chord)
        # print("h", h.shape)
        
        h = torch.cat([h, x_chord], dim=-1)  # 在global sos的基础上cat， 在进入model之前
        h = self.fuse(h)
        # Prepend SOS token and remove the last token
        sos = self.global_sos.view(1, 1, -1).repeat(batch_size, 1, 1)
        h = torch.cat([sos, h[:, :-1]], dim=1)

       
        h = self.model(h, attention_mask=self.buffered_future_mask(h), interleave_pos=True)[0] ##all the sos of every timestep (considering other timestep)
        return self.local_decode(h, emb)


        
    def preprocess(
        self,
        x: torch.LongTensor, # melody
        pitch_shift: torch.LongTensor,
        x_chord: Optional[torch.LongTensor] = None,
        chord_pitch_shift:Optional[torch.LongTensor] = None,
        y: Optional[torch.LongTensor] = None, # accompaniment
    ):

        batch_size, seq_length, subseq_length = x.shape
        x = x.long().view(batch_size, seq_length, subseq_length // 3, 3)
        x_processed = torch.zeros(batch_size, seq_length, subseq_length // 3, 2, dtype=torch.long, device=x.device)
        pad_indices = x[:, :, :, 1] == 255 #pitch is 255 that need to be pad
        eos_indices = x[:, :, :, 0] == 254 #program is 254
        is_not_drum = x[:, :, :, 0] != 127 
        x_processed[:, :, :, 0] = 0 # program 不变
        x_processed[:, :, :, 1] = x[:, :, :, 1] + (x[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum
        x_processed[pad_indices] = PAD_TOKEN
        x_processed[:, :, :, 0][eos_indices] = EOS_TOKEN

        for b in range(batch_size):
            x_chord[b] = x_chord[b].roll(shifts=chord_pitch_shift[b].item(), dims=-1)        
        x_chord = x_chord.repeat_interleave(2, dim=1).long()
        if y==None:
            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2), x_chord
        else:
            batch_size_y, seq_length_y, subseq_length_y = y.shape
            y = y.long().view(batch_size_y, seq_length_y, subseq_length_y // 3, 3)
            y_processed = torch.zeros(batch_size_y, seq_length_y, subseq_length_y // 3, 2, dtype=torch.long, device=y.device)
            pad_indices_y = y[:, :, :, 1] == 255 #pitch is 255 that need to be pad
            eos_indices_y = y[:, :, :, 0] == 254 #program is 254
            is_not_drum_y = y[:, :, :, 0] != 127 
            y_processed[:, :, :, 0] = 1 # program 不变
            y_processed[:, :, :, 1] = y[:, :, :, 1] + (y[:, :, :, 2]) * 128 + 2 + pitch_shift[:, None, None] * is_not_drum_y
            y_processed[pad_indices_y] = PAD_TOKEN
            y_processed[:, :, :, 0][eos_indices_y] = EOS_TOKEN

            return x_processed.view(batch_size, seq_length, subseq_length // 3 * 2), y_processed.view(batch_size_y, seq_length_y, subseq_length_y // 3 * 2), x_chord

    
    def loss(self, x_mel, x_acc, x_chord, pitch_shift, chord_pitch_shift):
        # x_mel, x_acc = self.preprocess(x_mel, pitch_shift, y = x_acc)
        # print(x_mel==x_acc)
        x_mel, x_acc, x_chord = self.preprocess(x_mel, pitch_shift, x_chord, chord_pitch_shift, y=x_acc)
        print(x_mel.shape)
        print(x_acc.shape)
        for i in range(x_mel.shape[0]):
            decode_output(x_mel[i], f'samples/{i}_mel.mid', single=True)
            decode_output(x_acc[i], f'samples/{i}_acc.mid', single=True)
            chroma_to_midi(x_chord[i], f'samples/{i}_cho.mid', base_pitch=48)

        pdb.set_trace()
        batch_size, seq_len, subseq_len = x_mel.shape # 10*384*8
        stacked = torch.stack([x_acc, x_mel], dim=2)
        x = stacked.view(batch_size, seq_len * 2, subseq_len)
        
        x_target = x.clone()
        # build a mask: True at every odd timestep
        idx = torch.arange(seq_len * 2, device=x.device)
        mel_mask = (idx % 2 == 1).unsqueeze(0).unsqueeze(-1)    # [1, 2*S, 1]
        mel_mask = mel_mask.expand(batch_size, seq_len * 2, subseq_len)                 # [B, 2*S, L]
        x_target[mel_mask] = PAD_TOKEN

        if x.device.type == 'cuda':
            torch.cuda.synchronize()            # flush pending work
        t0 = time.perf_counter()

        y = self(x, x_chord)

        if x.device.type == 'cuda':
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - t0) * 1000.0
        print(forward_ms)
        self.log('fwd_latency_ms', forward_ms, prog_bar=True, on_step=True, on_epoch=False)

        # y = self(x, x_chord)

        return F.cross_entropy(y.view(-1, N_TOKENS), x_target.view(-1), ignore_index=PAD_TOKEN)

    def training_step(self, batch, batch_idx):
        x_mel, x_acc, _, _, _ = batch
        # print(x_mel==x_acc)
        loss = self.loss(*batch)
        self.log('train_loss', loss)
        # scheduler step
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log('training/lr', scheduler.get_last_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        max_lr = 1e-4
        optimizer = torch.optim.AdamW(self.parameters(), lr=max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=MAX_STEPS, pct_start=0.005)
        return [optimizer], [scheduler]

class FramedDataset(IterableDataset):

    def __init__(self, file_path, target_length, batch_size, split='all', split_ratio=10):
        self.file_path = file_path
        self.length = torch.load(file_path[:-3] + '.length.pt', weights_only=True) # 每个歌的长度
        self.start = torch.cumsum(self.length, dim=0) - self.length # 累计 -- 每首歌开始的地方
        # Invalid samples are those whose length is less than min_length
        is_valid = self.length >= target_length
        self.valid_indices = torch.arange(len(self.start))[is_valid]
        # pdb.set_trace()
        # Get training or validation split
        if split == 'all':
            pass
        elif split == 'train':
            self.valid_indices = self.valid_indices[self.valid_indices % split_ratio != 0]
        elif split == 'val':
            self.valid_indices = self.valid_indices[self.valid_indices % split_ratio == 0]
        self.split = split
        self.valid_song_count = len(self.valid_indices)
        self.target_length = target_length
        self.batch_size = batch_size
        print('Metadata for dataset', file_path, 'loaded. Number of valid songs:', self.valid_song_count)

    def __iter__(self):
        data = torch.load(self.file_path, weights_only=True) #伴奏
        data_c = torch.load(self.file_path.replace('acc', 'mel'), weights_only=True) #旋律
        print(data_c==data)
        data_chord = torch.load(self.file_path.replace('acc', 'chord'), weights_only=True)

        pitch_shift_range = torch.load(self.file_path[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
        pitch_shift_range[pitch_shift_range[:, 0] < -5, 0] = -5
        pitch_shift_range[pitch_shift_range[:, 1] > 6, 1] = 6
        pitch_shift_range_c = torch.load(self.file_path.replace('acc', 'mel')[:-3] + '.pitch_shift_range.pt', weights_only=True).reshape(-1, 2)
        pitch_shift_range_c[pitch_shift_range_c[:, 0] < -5, 0] = -5
        pitch_shift_range_c[pitch_shift_range_c[:, 1] > 6, 1] = 6
        

        if self.split == 'val':
            pitch_shift_range = torch.zeros_like(pitch_shift_range)  # No pitch shift for validation
            pitch_shift_range_c = torch.zeros_like(pitch_shift_range_c)  # No pitch shift for validation
        print('Data for dataset', self.file_path, 'loaded.')
        while True:
            indices = torch.randperm(len(self.valid_indices))
            for i in range(0, len(self.valid_indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_pitch_shift_range = pitch_shift_range[self.valid_indices[batch_indices]]
                batch_pitch_shift_range_c = pitch_shift_range_c[self.valid_indices[batch_indices]]
                raw_ids = self.valid_indices[batch_indices]
                # 'start' of each song is not necessarily a multiple of 8 because song length is not necessailty divisible by 1 bar
                # BUT the 'start' should always be downbeat (even when it's empty)
                # we just need to make sure whatevery is adding on to it is a multiple of 8 
                to_be_added = torch.floor(torch.rand(len(raw_ids)) * (self.length[raw_ids] - self.target_length)).long()
                to_be_added -= to_be_added % 8
                # print("-1", to_be_added%8)
                starts = to_be_added + self.start[raw_ids] # 在每首歌的开始时间上 加上（整首歌的长度-target_length）
                print('starts', starts)
                index_matrix = torch.arange(self.target_length).view(1, -1) + starts.view(-1, 1)
                minmax = torch.minimum(batch_pitch_shift_range_c[:, 1], batch_pitch_shift_range[:, 1])
                maxmin = torch.maximum(batch_pitch_shift_range[:, 0], batch_pitch_shift_range_c[:, 0])
                batch_pitch_shift = torch.floor(torch.rand(len(raw_ids)) * (minmax - maxmin + 1)).long() + maxmin
                chord_pitch_shift = batch_pitch_shift % 12
                # print("0", raw_ids) # 10 个id （代表第几首歌）
                # print("1", starts) # n * 
                # print("2", self.target_length) # 192？？
                # print("3", index_matrix)


                yield data_c[index_matrix], data[index_matrix], data_chord[index_matrix], batch_pitch_shift, chord_pitch_shift

def sanity_check():
    # Do some sanity check
    model = RoFormerSymbolicTransformer()
    x = [
        [
            [0 , 34, 12, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [12, 34, 52, 34, 12, EOS_TOKEN],
            [12, 34, 52, 34, 12, 52]
        ],
        [
            [52, 34, 12, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [12, 34, 52, 34, 12, EOS_TOKEN],
            [EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
        ]
    ]
    y = [
        [
            [42, 34, 12, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [22, 34, 52, 34, 12, EOS_TOKEN],
            [22, 34, 52, 34, 12, 52]
        ],
        [
            [42, 34, 12, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN],
            [22, 34, 52, 34, 12, EOS_TOKEN],
            [EOS_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
        ]
    ]
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    loss = model.loss(x, y, pitch_shift = torch.zeros(1, dtype=torch.int8))
    # print(loss)

if __name__ == '__main__':
    # sanity_check()
   
    parser = argparse.ArgumentParser(description="process midi folder(s) into usable tensors for the task")
    
    parser.add_argument("--batch_size",type=int, default=10, help="batch size")
    parser.add_argument("--model_size", type=str, default='small', help="large or small")
    parser.add_argument("--path_to_dataset",type=str,help="name to your accompaniment .pt file")
    parser.add_argument("--model_name",type=str,default=None, help="name to your model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="continue training from checkpoint")
    parser.add_argument("--wandb", action="store_true",  default=False, help="enable wand logging")

    args = parser.parse_args()

    batch_size = args.batch_size 
    model_size = args.model_size
    dataset = args.path_to_dataset
    checkpoint_path = args.checkpoint_path

    assert model_size in ['small', 'large']
    n_gpus = max(torch.cuda.device_count(), 1)

    default_name = f"m2a_transformer_v0.4_chord_{model_size}_batch_{batch_size * n_gpus}_schedule"
    model_name = args.model_name if args.model_name is not None else default_name
    net = RoFormerSymbolicTransformer(model_size == 'large')
    train_set_loader = DataLoader(FramedDataset(dataset, TRAIN_LENGTH, batch_size, split = 'train'), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(dataset, TRAIN_LENGTH, batch_size, split = 'val'), batch_size=None, num_workers=1, persistent_workers=True)
    checkpoint_callback = L.callbacks.ModelCheckpoint(monitor='val_loss',
                                                      save_top_k=10,
                                                      save_last=True,
                                                      dirpath=f'ckpt/{model_name}',
                                                      filename=model_name + '.{epoch:02d}.{val_loss:.5f}')

    trainer = L.Trainer(devices=n_gpus,
                        precision="bf16-mixed" if torch.cuda.is_available() else 32,
                        max_steps=MAX_STEPS,
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        callbacks=[checkpoint_callback],
                        val_check_interval=500,
                        limit_val_batches=25,
                        check_val_every_n_epoch=None,
                        logger= WandbLogger(name=model_name, project="StreamMUSE", 
                                config={
                                    "batch_size": batch_size,
                                    "model_size": model_size,
                                    "train_length": TRAIN_LENGTH
                                }) if args.wandb else TensorBoardLogger("tb_logs", name=model_name),
                        strategy='auto' if n_gpus == 1 else 'ddp')
    trainer.fit(net, train_set_loader, val_set_loader, ckpt_path=checkpoint_path)
    torch.save(net.state_dict(), f'ckpt/{model_name}.pt')