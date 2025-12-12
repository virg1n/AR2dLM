import time
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from datatrove.utils.dataset import DatatroveFolderDataset
import torch.distributed as dist


@dataclass
class TrainerConfig:
    vocab_size: int                 
    num_epochs: int                 

    use_ddp: bool  
    adap_factor: int
    masked_token_id: int
    pad_token_id: int
    
    clean_cuda_cache: bool = True   # Helps prevent OOM errors during eval on large models
    use_compile: bool = True        # use torch.compile()
    use_dtype: str = "bfloat16"

    seed: int = 1998                
    max_seq_len: int = 2048         # maximum context length for batch
    batch_size: int = 1             # numbe of batches
    accumulation_steps: int = 1
    
    # Optimizer parameters
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.90, 0.95)

    val_ratio: int = 0.005
    steps_for_eval: int = 20                            # number of steps for evaluation
    eval_interval: int = 200

    checkpoints_frequency: int = 2_000
    path_to_checkpoints: str = "./model_testing"        # path to directory to save checkpoints

    tokenized_dataset_path: str = ""                    # path to directory with tokeized dataset
    eval_log_file: str = "logs/eval.txt"                # path to file to write eval results

    



class DataLoader():
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.current_epoch = 0
        self.seed = config.seed
        self.token_size = 2 if config.vocab_size < 65535 else 4
        self.rank = rank

        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        if rank == 0:
            print(f"{'Total tokens loaded: '} {self.len_dataset * config.max_seq_len:,}")

        self.train_len_dataset = math.ceil((1-config.val_ratio) * self.len_dataset)
        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        # shard_size = self.len_dataset // world_size 
        # # self.train_start_idx = rank * shard_size
        # # self.train_end_idx = self.train_start_idx + shard_size
        # # self.train_current_idx = self.train_start_idx

        # # self.val_start_idx = self.train_len_dataset
        # # self.val_current_idx = self.val_start_idx
        # self.train_start_idx = 0
        # self.train_end_idx = self.train_len_dataset
        # self.train_current_idx = self.train_start_idx
        
        # self.val_start_idx = self.train_len_dataset
        # self.val_current_idx = self.val_start_idx
        shard_size = math.ceil(self.train_len_dataset / world_size)

        self.train_start_idx = rank * shard_size
        self.train_end_idx = min(self.train_start_idx + shard_size, self.train_len_dataset)
        self.train_current_idx = self.train_start_idx

        # validation: everyone sees the full validation set (simple & fine),
        # or you could also shard similarly if you prefer.
        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx


    def get_batch(self, current_idx: int, start_idx: int, end_idx: int):
        new_idx = current_idx + self.config.batch_size
        
        x_l = [
            self.dataset[idx]['input_ids']
            for idx in range(current_idx, min(new_idx, end_idx))
        ]
        x = torch.stack(x_l)
    
        if new_idx >= end_idx:
            new_idx = start_idx
            self.new_epoch()

        return x, new_idx

    def next_batch(self, split):
        if split == "train":
            x, self.train_current_idx = self.get_batch(self.train_current_idx, self.train_start_idx, self.train_end_idx)
        else: # validation
            x, self.val_current_idx = self.get_batch(self.val_current_idx, self.val_start_idx, self.len_dataset)
        return x
    
    def reset(self, rank: int = 0, world_size: int = 1):
        self.current_epoch = 0
        self.seed = self.config.seed
        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = math.ceil(self.train_len_dataset / world_size)
        self.train_start_idx = rank * shard_size
        self.train_end_idx = min(self.train_start_idx + shard_size, self.train_len_dataset)
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def new_epoch(self):
        self.current_epoch += 1
        self.load_dataset(self.seed + self.current_epoch)

    def load_dataset(self, seed: int):
        self.dataset = DatatroveFolderDataset(
            folder_path=self.config.tokenized_dataset_path,  # e.g. "./fwe-10BT"
            filename_pattern="*.ds",                         # pattern *inside* that folder
            seq_len=self.config.max_seq_len,
            token_size=self.token_size,
            recursive=False,                                # your .ds files are in the top level
            shuffle=True,
            seed=seed + self.rank,
        )

    def num_train_steps(self):
        return math.ceil((self.train_end_idx-self.train_start_idx) / self.config.batch_size)


class Trainer():
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.num_epochs = config.num_epochs
        self.masked_token_id = config.masked_token_id
        self.pad_token_id = config.pad_token_id

        self.adap_factor = config.adap_factor

        self.clean_cuda_cache = config.clean_cuda_cache
        self.dtype = getattr(torch, self.config.use_dtype)

        self.steps_for_eval = config.steps_for_eval
        self.weight_decay = config.weight_decay

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n_gpus = 0
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)
            n_gpus = torch.cuda.device_count()

        use_compile = self.config.use_compile and self.device.type == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)
            
        if n_gpus > 1 and config.use_ddp:
            self.ddp = True

            # --- NEW: initialize process group ---
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.ddp_rank = dist.get_rank()
            self.ddp_world_size = dist.get_world_size()
            # torchrun sets LOCAL_RANK; fall back to rank 0 if not present
            self.ddp_local_rank = int(os.environ.get("LOCAL_RANK", self.ddp_rank))

            self.device = torch.device(f"cuda:{self.ddp_local_rank}")
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0

            self.model.to(self.device)

            # Wrap with DDP on the local device
            self.model = DDP(
                self.model,
                device_ids=[self.ddp_local_rank],
                output_device=self.ddp_local_rank,
                find_unused_parameters=False,
            )
            self.raw_m = self.model.module  # underlying model
        else:
            self.ddp = False
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.master_process = True

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device.type != "cpu":
                self.model.to(self.device)

        if self.master_process:
            print("Device:", self.device)
            print(f"Model's trainable params: {sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) / 1e6:.2f}M")
            print(f"Tokens per step: {self.config.batch_size * self.config.max_seq_len * self.ddp_world_size * self.config.accumulation_steps}")
            print(f"use {'torch.compile()'}: {use_compile}")
           
           
    def _apply_mask_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Apply discrete absorbing diffusion noise:
        with prob alpha_t keep token, with prob (1 - alpha_t) replace by [MASK].
        """
        mask_token_id = self.masked_token_id
        pad_token_id = self.pad_token_id

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=x0.device)
        else:
            t = t.to(dtype=torch.float32, device=x0.device)

        if t.dim() == 0:
            alpha_t = 1.0 - t
            alpha_t_expanded = alpha_t
            noise = torch.rand_like(x0, dtype=torch.float32)
        elif t.dim() == 1:
            alpha_t = 1.0 - t
            noise = torch.rand_like(x0, dtype=torch.float32)
            alpha_t_expanded = alpha_t.view(-1, 1) 
        else:
            raise ValueError("t must be scalar or shape (batch,)")

        # Decide which tokens to keep
        if t.dim() == 0:
            keep = noise < alpha_t_expanded
        else:
            keep = noise < alpha_t_expanded

        xt = x0.clone()

        xt[~keep] = mask_token_id

        if pad_token_id is not None:
            pad_mask = (x0 == pad_token_id)
            xt[pad_mask] = pad_token_id

        return xt

    
    def _get_mask_for_step(self, seq_len: int, progress: float):
        """
        Boolean attention mask that anneals from causal to full attention.

        Returns:
            attn_mask: (1, 1, seq_len, seq_len) bool tensor
            attn_mask[..., i, j] == True  -> position i is allowed to attend to j
            attn_mask[..., i, j] == False -> position i is NOT allowed to attend to j
        """
        # how many positions to the right each token can see
        max_extra_right = seq_len - 1
        extra_right = int(math.floor(progress * max_extra_right))

        idx = torch.arange(seq_len, device=self.device)
        q = idx.view(-1, 1)   # (L, 1)
        k = idx.view(1, -1)   # (1, L)

        allowed = k <= (q + extra_right)   # (L, L) bool

        # add batch/head dims: (1, 1, L, L)
        attn_mask = allowed.unsqueeze(0).unsqueeze(0)
        return attn_mask
    

    def step(self, data_loader, accumulation_steps: int,
              num_tokens: int, tau: torch.Tensor, progress: float,            
              split: str = "train"):
        """
        Performs single forward/backward pass with gradient accumulation.
            Returns: (loss, number_of_processed_tokens)
        """
        x0 = data_loader.next_batch(split=split)
        x0 = x0.to(self.device)
        xt = self._apply_mask_noise(x0, tau)

        num_tokens += torch.numel(x0)

        attn_mask = self._get_mask_for_step(seq_len=xt.size(1), progress=progress)
        masked_positions = (xt == self.masked_token_id)
        if self.pad_token_id is not None:
            masked_positions &= (x0 != self.pad_token_id)
        
        # Avoid no-mask cases
        if masked_positions.sum() == 0:
            return torch.tensor(0.0, device=self.device), num_tokens
        masked_positions[:, 0] = False

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            _, loss = self.model(xt, x0, attn_mask, masked_positions)
            print(f"loss_before_divide: {loss}")
            loss = loss / tau.to(loss.dtype)


        loss = loss / accumulation_steps

        loss.backward()
        return loss.detach(), num_tokens

    # def step(self, data_loader, accumulation_steps: int,
    #      num_tokens: int, tau: torch.Tensor, progress: float,
    #      split: str = "train"):
    
    #     x0 = data_loader.next_batch(split=split).to(self.device)
    #     xt = self._apply_mask_noise(x0, tau)
    
    #     num_tokens += torch.numel(x0)
    
    #     attn_mask = self._get_mask_for_step(seq_len=xt.size(1), progress=progress)
    #     masked_positions = (xt == self.masked_token_id)
    #     if self.pad_token_id is not None:
    #         masked_positions &= (x0 != self.pad_token_id)
    
    #     if masked_positions.sum() == 0:
    #         return torch.tensor(0.0, device=self.device), num_tokens
    #     masked_positions[:, 0] = False
    
    #     with torch.autocast(device_type=self.device.type, dtype=self.dtype):
    #         # raw CE loss
    #         _, raw_loss = self.model(xt, x0, attn_mask, masked_positions)
    
    #     # diffusion weighting 1/t
    #     weighted_loss = raw_loss / tau.to(raw_loss.dtype)
    
    #     # scale for grad accumulation
    #     loss = weighted_loss / accumulation_steps
    #     loss.backward()
    
    #     return raw_loss.detach(), num_tokens

    def train(self, data_loader):
        num_steps_per_epoch = math.ceil(data_loader.num_train_steps() / self.config.accumulation_steps)

        # Configuration of optimizer and schedulers
        # Using AdamW with cosine decay and warmup - similar to Llama's training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),  
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.weight_decay,
            fused=(self.device.type=="cuda")
        )
        
        warmup_steps = math.floor(self.config.warmup_ratio * num_steps_per_epoch * self.num_epochs)
        warmup_factor = lambda step: 0.05 + 0.95 * (step / max(warmup_steps, 1))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_factor
        )

        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(num_steps_per_epoch * self.num_epochs) - warmup_steps, 
            eta_min=0.1 * self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_steps])

        last_step = num_steps_per_epoch - 1
        self.model.train()
        eps = 5e-4

        print("TOTAL STEPS: ", num_steps_per_epoch * self.num_epochs)

        for epoch in range(self.num_epochs):
            for step in range(num_steps_per_epoch):
                progress = min(step/(num_steps_per_epoch * self.adap_factor) + int(epoch), 1.0)

                tau = torch.rand(1, device=self.device)
                tau = tau.clamp(min=eps)

                t0 = time.perf_counter()
                accumulated_loss = 0.0
                num_tokens = 0

                ddp_nosync_ctx = self.model.no_sync() if self.ddp else nullcontext()
                with ddp_nosync_ctx:
                    for _ in range(self.config.accumulation_steps - 1):
                        loss, num_tokens = self.step(data_loader, self.config.accumulation_steps, num_tokens, tau, progress, split="train")
                        accumulated_loss += float(loss)

                loss, num_tokens = self.step(
                    data_loader, self.config.accumulation_steps,
                    num_tokens, tau, progress, split="train"
                )
                accumulated_loss += float(loss)
                
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                t1 = time.perf_counter()

                tokens_per_sec = num_tokens / (t1 - t0) * self.ddp_world_size

                # Logging 
                if self.master_process:
                    print(f"Epoch: {epoch} | Step: {step} |  loss: {accumulated_loss:.4f} | norm: {norm:.4f} | lr: {scheduler.get_last_lr()[0]} | tok/s: {tokens_per_sec} | progress: {progress}")
                    with open("train_loss.txt", "a") as f:
                        f.write(f"Epoch: {epoch} | Step: {step} |  loss: {accumulated_loss:.4f} | norm: {norm:.4f} | lr: {scheduler.get_last_lr()[0]} | tok/s: {tokens_per_sec} | progress: {progress}\nw")
                
                # Evaluation 
                if self.master_process and ((step>0 and step % self.config.eval_interval == 0) or step == last_step):
                    self.model.eval() 
                    val_loss = self.eval(data_loader)

                    with open(self.config.eval_log_file, "a") as f:
                        f.write(f"Step: {step * (epoch+1)}, val_loss: {val_loss.item():.4f}, norm: {norm:.4f}, lr: {scheduler.get_last_lr()[0]}, time: {t1 - t0:.2f}s, tok/s: {tokens_per_sec:.1f} \n")

                    

                    self.model.train()
                    if self.clean_cuda_cache:
                        torch.cuda.empty_cache()

                # Save Chekpoints
                if self.master_process and ((step % self.config.checkpoints_frequency == 0 and step > 0) or step == last_step):
                    self.save_checkpoints(optimizer, self.config.path_to_checkpoints, name=str((epoch+1) * step))
        if self.ddp:
            dist.destroy_process_group()
    def eval(self, data_loader):
        """
        Evaluates model on validation split using running average of
        first [steps_for_eval] batches under the same diffusion objective.
        """
        eps = 5e-4
        val_loss_accum = 0.0
        effective_steps = 0

        with torch.no_grad():
            for _ in range(self.steps_for_eval):
                # 1) Get a validation batch
                x0 = data_loader.next_batch(split="val").to(self.device)

                # 2) Sample time and corrupt with absorbing noise
                tau = torch.rand(1, device=self.device)
                tau = tau.clamp(min=eps)
                xt = self._apply_mask_noise(x0, tau)

                # 3) Full bidirectional attention at eval
                attn_mask = self._get_mask_for_step(
                    seq_len=xt.size(1),
                    progress=1.0,    # fully annealed mask
                )

                # 4) Masked positions = where xt == [MASK] (excluding pad)
                masked_positions = (xt == self.masked_token_id)
                if self.pad_token_id is not None:
                    masked_positions &= (x0 != self.pad_token_id)

                # It *can* happen there are no masked tokens for very small tau;
                # skip those batches so we don't step on stale gradients.
                if masked_positions.sum() == 0:
                    continue

                # 5) Forward + loss, scaled by 1/tau
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, loss = self.model(xt, x0, attn_mask, masked_positions)
                    loss = loss / tau.to(loss.dtype)

                val_loss_accum += loss.detach().float()
                effective_steps += 1

        if effective_steps == 0:
            # extremely unlikely, but just in case
            return torch.tensor(0.0, device=self.device)

        return val_loss_accum / effective_steps


    # def save_checkpoints(self, optimizer, path: str, name: str):
    #     os.makedirs(path, exist_ok=True)
    #     checkpoint_path = os.path.join(path, f"model.checkpoint.{name}.pt")
    #     # self.model.save_pretrained(".checkpoint_path", config=config)
    #     checkpoint = {
    #                 'model': self.model.state_dict(),
    #                 'optimizer':optimizer.state_dict(),
    #             }
    #     torch.save(checkpoint, checkpoint_path)
    #     print("Checkpoints saved")
    def save_checkpoints(self, optimizer, path: str, name: str):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model.checkpoint.{name}.pt")
    
        if self.ddp:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
    
        checkpoint = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print("Checkpoints saved")