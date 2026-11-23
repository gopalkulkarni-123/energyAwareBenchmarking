import torch
d, bf = "cuda", torch.bfloat16
a = torch.randn(4096, 4096, dtype=bf, device=d)
b = torch.randn(4096, 4096, dtype=bf, device=d)
w = torch.randn(2560, 9728, dtype=bf, device=d)
v = torch.randn(1, 2560, dtype=bf, device=d)
x = torch.randn(256 * 1024 * 1024, dtype=bf, device=d)
torch.cuda.synchronize()
torch.cuda.profiler.start()
c = a @ b          # big matmul: 2 x 4096^3 = 137.4 GFLOP
u = v @ w          # decode-like: 2 x 2560 x 9728 = 49.8 MFLOP, reads 49.8 MB of weights
y = x.clone()      # copy: reads 536.9 MB, writes 536.9 MB
torch.cuda.synchronize()
torch.cuda.profiler.stop()