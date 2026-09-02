## 01_gemm_shapes

Output
```text
torch 2.14.0+cu130 on NVIDIA RTX 6000 Ada Generation

               shape  dtype         ms      GFLOP/s  FLOP/byte
--------------------------------------------------------------
   prefill, seq=4096   fp32     7.8843      17432.0      682.7
   prefill, seq=4096   fp16     2.3888      57533.8     1365.3
   prefill, seq=4096   bf16     1.8215      75453.6     1365.3

    prefill, seq=512   fp32     1.1614      14792.8      204.8
    prefill, seq=512   fp16     0.2759      62264.7      409.6
    prefill, seq=512   bf16     0.2305      74548.8      409.6

    decode, batch=32   fp32     0.0546      19654.7       15.8
    decode, batch=32   fp16     0.0247      43509.4       31.5
    decode, batch=32   bf16     0.0196      54899.3       31.5

     decode, batch=1   fp32     0.0823        407.8        0.5
     decode, batch=1   fp16     0.0523        641.3        1.0
     decode, batch=1   bf16     0.0841        399.1        1.0
```

Takeaways
- When the sequence length is high (4096, 512), we are simulating prefill, where we have a lot of tokens we can compute
- When the the number of tokens processed together is low (32, 1), we are simulating decode
- As dtype gets smaller, less bytes get moved, so arithmetic density rises (FLOPS/byte) and execute faster (GFLOPS/s)
- Comparing sequence lengths, we can see that arithmetic intensity (FLOPS/byte) is higher with higher seq lengths
- We can batch decode, but that doesn't fully close the gap (50k GFLOP/s vs 75k GFLOPS/s)
