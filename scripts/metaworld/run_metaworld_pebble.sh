echo $@

env=$1
teacher_eps_skip=$2
device=$3
seed=$4

reward_batch=100
feed_type=1
skip_impl=mean_traj_error


python finetune_pb.py \
    task=metaworld_${env}-v2 \
    agent=apt_sac_dev_metaworld \
    snapshot_ts=10000 \
    use_wandb=true \
    seed=$seed \
    agent.actor_lr=0.0003 \
    agent.critic_lr=0.0003  \
    num_seed_frames=0 \
    num_train_frames=1500000 \
    agent.batch_size=512 \
    agent.critic_cfg.hidden_dim=256 \
    agent.critic_cfg.hidden_depth=3 \
    agent.actor_cfg.hidden_dim=256 \
    agent.actor_cfg.hidden_depth=3 \
    device=$device \
    num_interact=5000 max_feedback=50000 \
    reward_batch=$reward_batch reward_model_train_epoch=50 \
    feed_type=$feed_type \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0 \
    teacher_eps_skip=$teacher_eps_skip \
    teacher_eps_equal=0 \
    skip_impl=$skip_impl \
    agent.init_all=false \
    save_video=true \
