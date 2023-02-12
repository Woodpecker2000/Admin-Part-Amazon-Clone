
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import pyarrow as pa

from pathlib import Path

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    arrow_dir = Path(ckpt_dir).expanduser() / 'arrow'

    if not arrow_dir.exists():
        print('Converting checkpoints to arrow format')
        checkpoints = sorted(Path(ckpt_dir).expanduser().glob("*.pth"))
        for ckpt_file in checkpoints:
            print(ckpt_file)
            index = ckpt_file.parts[-1].split('.')[-2]

            ckpt = torch.load(ckpt_file, map_location='cpu')
            (arrow_dir / index).mkdir(parents=True, exist_ok=True)
            for k, v in ckpt.items():
                tens = pa.Tensor.from_numpy(v.numpy())
                with pa.output_stream(arrow_dir / index / k) as f:
                    pa.ipc.write_tensor(tens, f)
            ckpt = None

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    print("Loading checkpoint")
    segments = sorted((arrow_dir / '00').glob("*"))
    # print(segments)

    checkpoint = {}
    files = []
    for seg in segments:
        f = pa.memory_map(str(seg))
        files.append(f)
        t = pa.ipc.read_tensor(f).to_numpy()
        t = torch.from_numpy(t)
        checkpoint[seg.parts[-1]] = t

    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.FloatTensor)

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    print("Loading tokenizer")
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    print("Loading model")
    model = Transformer(model_args)

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    model.load_state_dict(torch.load(checkpoints[-1]), strict=False)

    for f in files:
        f.close()
    files = None

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")