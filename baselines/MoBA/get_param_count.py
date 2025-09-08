import torch
from model import OmniBioTA, OmniBioTAConfig
import argparse

if __name__ == "__main__":
    dtype = torch.half
    device = "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--ctx_len", type=int, default=1024)
    
    args = parser.parse_args()

    config = OmniBioTAConfig()
    config.vocab_size = 4096
    config.block_size = args.ctx_len
    config.n_embd = args.n_embd
    config.n_layer = args.n_layer
    config.n_head = args.n_head
    config.flash = True

    model = OmniBioTA(config)
    print(model.get_num_params())

    actual_flops = 6 * model.get_num_params() + 12 * config.n_layer * config.n_embd * args.ctx_len # 6N + 12*LHQT estimate of flops per token
    optimal_training_tokens = model.get_num_params()*20
    actual_flops *= 180e9 # optimal_training_tokens
    gpu_flops_per_sec = 1671e12*8 # A100 flops per second 312e12, H100 flops per second 1671e12
    estimated_training_time = actual_flops / gpu_flops_per_sec
    ETT_hours = estimated_training_time / 3600
    efficiency = 0.2
    ETT_hours /= efficiency # Estimated efficiency for of our code for GPUs
    print("Estimate Training Time: %.2f hours" % (ETT_hours))

    actual_flops = 6 * model.get_num_params() + 4 * config.n_embd ** 2 + 12 * config.n_layer * config.n_embd * 11400 + 12 * config.n_layer * config.n_embd * 900 # 6N + 12*LHQT estimate of flops per token
    optimal_training_tokens = model.get_num_params()*20
    actual_flops *= 180e9 # optimal_training_tokens
    gpu_flops_per_sec = 312e12 # A100 flops per second 312e12, H100 flops per second 1671e12
    estimated_training_time = actual_flops / gpu_flops_per_sec
    ETT_hours = estimated_training_time / 3600
    efficiency = 0.2
    ETT_hours /= efficiency # Estimated efficiency for of our code for GPUs
    print("Estimate Training Time: %.2f hours" % (ETT_hours))

    actual_flops = 6 * model.get_num_params() + 12 * config.n_layer * config.n_embd * 150_000 # 6N + 12*LHQT estimate of flops per token
    optimal_training_tokens = model.get_num_params()*20
    actual_flops *= 180e9 # optimal_training_tokens
    gpu_flops_per_sec = 312e12 # A100 flops per second 312e12, H100 flops per second 1671e12
    estimated_training_time = actual_flops / gpu_flops_per_sec
    ETT_hours = estimated_training_time / 3600
    efficiency = 0.2
    ETT_hours /= efficiency # Estimated efficiency for of our code for GPUs
    print("Estimate Training Time: %.2f hours" % (ETT_hours))