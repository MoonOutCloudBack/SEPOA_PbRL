echo $@

agent=$1  # aps_sac, cic_sac, diayn_sac
task=$2
max_feedback=$3
reward_batch=$4
teacher_eps_skip=$5
seed=$6
device=$7
snapshot_ts=$8  # 500000 or 1000000

python finetune_pb_sd_clean_skills.py \
    agent=$agent \
    task=$task \
    agent.lr=0.0005 \
    seed=$seed \
    num_seed_frames=0 \
    num_train_frames=1000000 \
    use_wandb=true \
    device=$device \
    num_interact=20000 max_feedback=$max_feedback \
    reward_batch=$reward_batch reward_model_train_epoch=50 \
    feed_type=8 \
    teacher_beta=-1 \
    teacher_gamma=1 \
    teacher_eps_mistake=0 \
    teacher_eps_skip=$teacher_eps_skip \
    teacher_eps_equal=0 \
    skip_impl="mean_traj_error" \
    snapshot_ts=$snapshot_ts \
    agent.init_all=true \

