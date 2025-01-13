echo $@

python pretrain_hx.py \
    domain=$1 \
    agent=apt_sac_dev_metaworld \
    use_wandb=false \
    seed=$3 \
    agent.lr=0.0005 \
    num_seed_frames=1000 \
    num_train_frames=10000 \
    snapshots=[10000] \
    agent.actor_lr=0.0003 \
    agent.critic_lr=0.0003  \
    agent.batch_size=512 \
    agent.critic_cfg.hidden_dim=256 \
    agent.critic_cfg.hidden_depth=3 \
    agent.actor_cfg.hidden_dim=256 \
    agent.actor_cfg.hidden_depth=3 \
    device=$2 \
        
