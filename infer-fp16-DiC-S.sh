# sample_ddp.py 是 分布式采样脚本，里面有：dist.init_process_group("nccl")
# 它默认用 env:// 初始化方式，所以 必须 由 torchrun（或 python -m torch.distributed.run）之类的启动器来跑，启动器会自动给每个进程设置 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT 等环境变量。
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp.py --ckpt="./results/000-DiC-S/checkpoints/0050000.pt" --image-size=256 --model=DiC-S --cfg-scale=1