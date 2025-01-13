echo $@

python pretrain_hx.py \
    agent=aps_sac_dev_metaworld \
    domain=$1 \
    seed=$3 \
    use_wandb=true \
    num_train_frames=1000010 \
    agent.ent_reward_ratio=5 \
    device=$2 \

