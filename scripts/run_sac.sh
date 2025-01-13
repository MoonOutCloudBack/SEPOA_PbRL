echo $@

python finetune_hx_buffer.py \
    agent=sac \
    task=$1 \
    agent.lr=0.0003 \
    agent.nstep=1 \
    seed=$3 \
    snapshot_ts=0 \
    num_seed_frames=20000 \
    num_train_frames=1000000 \
    use_wandb=true \
    device=$2 \

