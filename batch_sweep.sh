for B in 4 8 16 32; do
  python dvfs_probe.py sweep --gpu 0 --caps 150,225,300 --clocks 510,810,1110,1410 --reps 2 \
    --workload "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bench_vllm.py --model ./qwen3_4b/ --bs $B --in-len 1024 --out-len 512 --warmup 1" \
    --out bs${B}_seq1024.csv
done
