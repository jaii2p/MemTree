# Running memtree on Vast.ai

## 1. Rent the machine
- **GPU:** 1x H100 80GB (or A100 80GB). Pick a host with high reliability.
- **Template:** a PyTorch template (CUDA + PyTorch preinstalled).
- **Disk:** 80 GB or more.
- **Type:** on-demand, not interruptible, for the long run.

## 2. Set up (about 5 min)
Upload `memtree.zip` (Jupyter upload, or `scp -P <port> memtree.zip root@<host>:~`), then:

    unzip memtree.zip && cd memtree-project
    pip install -r requirements.txt
    python run_tests.py                  # expect 0 failed, nothing skipped
    python scripts/selftest_phi.py       # 7 checks on real Phi-4 Mini (downloads ~8 GB)

If either one fails, stop and paste the output back to Claude before spending
more GPU time.

## 3. Get the books (about 5-15 min)

    python scripts/make_corpus.py --tokens 30M

This writes `corpus.txt` for training and `haystack.txt`, held-out text for
the recall test.

## 4. Quick run first (about 10 min)

    python -m memtree.distill --data corpus.txt --stage both --steps 300 \
        --seq 1024 --window 256 --out adapter_quick.pt

Check the `eval` lines: `kl` should go down and `top1_agree` up. The `stepN ...
(Ns)` lines give the real speed, so you can work out how long step 5 will take.

## 5. Serious run (a few hours)
Run it detached, so closing your browser or SSH doesn't kill it:

    nohup python -m memtree.distill --data corpus.txt --stage both --steps 6000 \
        --seq 2048 --window 256 --eval-every 500 --out adapter.pt > distill.log 2>&1 &
    tail -f distill.log                  # Ctrl+C stops watching, not training

`adapter.pt` is saved every 500 steps. If memory runs out in the KL stage,
add `--grad-ckpt`, or use `--seq 1024`.

## 6. Did the tree work?

    python scripts/eval_recall.py --adapter adapter.pt --haystack haystack.txt \
        --lengths 512 1024 2048 4096 --depths 0.1 0.5 0.9 --trials 10 | tee recall.txt

Compare **tree** against **window-only**, at lengths and depths where the fact
sits more than 256 tokens from the end. **original** is the ceiling.

## 7. Teach the protocol

    python -m memtree.finetune --adapter adapter.pt --synthetic 3000 \
        --steps 1500 --out finetune.pt | tee finetune.log

## 8. Before you destroy the instance
Download `adapter.pt`, `finetune.pt`, `distill.log`, `recall.txt` and
`finetune.log`. Instance storage is deleted with the machine.
