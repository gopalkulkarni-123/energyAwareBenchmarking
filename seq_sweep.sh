for L in 1024 4096 16384 ; do
  OUT=$((L/4)); [ $OUT -lt 512 ] && OUT=512
  python dvfs_probe.py sweep --gpu 0 --reps 5 \
    --caps 150,225,300 --clocks 510,810,1110,1410 \
    --workload "env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
      python bench_vllm.py --model ./qwen3_4b/ --bs 1 --in-len $L --out-len $OUT \
      --method double-len --warmup 1" \
    --out seq${L}_bs1.csv
done
