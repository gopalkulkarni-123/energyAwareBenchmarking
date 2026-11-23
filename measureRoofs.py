import torch, statistics as st

def timeit(fn, reps=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / 1000)        # seconds
    return st.median(ts)

d, bf = "cuda", torch.bfloat16
print(torch.cuda.get_device_name(0))

# bandwidth: copy 1 GiB (read 1 GiB + write 1 GiB)
x = torch.empty(2**29, dtype=bf, device=d)          # 2^29 x 2 bytes = 1 GiB
y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x))
bw = 2 * x.numel() * 2 / t / 1e12
print(f"copy:   {t*1e3:7.3f} ms  -> {bw:.3f} TB/s   ({bw/1.935*100:.0f}% of 1.935 TB/s)")
del x, y

# compute: bf16 matmul 8192^3
N = 8192
a = torch.randn(N, N, dtype=bf, device=d)
b = torch.randn(N, N, dtype=bf, device=d)
c = torch.empty(N, N, dtype=bf, device=d)
t = timeit(lambda: torch.matmul(a, b, out=c))
tf = 2 * N**3 / t / 1e12
print(f"matmul: {t*1e3:7.3f} ms  -> {tf:.1f} TFLOP/s ({tf/312*100:.0f}% of 312 TFLOP/s)")
print(f"measured knee: {tf/bw:.0f} FLOP per byte (paper: 161)")