echo $@

python pretrain_hx.py \
    agent=apt_sac \
    domain=$1 \
    seed=$3 \
    agent.lr=0.0005 \
    num_seed_frames=1000 \
    num_train_frames=10000 \
    snapshots=[10000] \
    device=$2 \

