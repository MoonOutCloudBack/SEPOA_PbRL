echo $@

feed_type=1

task=$1
teacher_eps_skip=$2
max_feedback=$3
device=$4
seed=$5

python finetune_pb.py \
    agent=apt_sac \
    task=$task \
    agent.lr=0.0005 \
    seed=$seed \
    snapshot_ts=10000 \
    num_seed_frames=0 \
    num_train_frames=1000000 \
    use_wandb=true \
    device=$device \
    num_interact=20000 max_feedback=$max_feedback \
    reward_batch=100 reward_model_train_epoch=50 \
    feed_type=$feed_type \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0 \
    teacher_eps_skip=$teacher_eps_skip \
    teacher_eps_equal=0 \
    skip_impl=mean_traj_error \
    agent.init_all=false \
    # data_aug_ratio=0 \  # uncomment it will become pebble

