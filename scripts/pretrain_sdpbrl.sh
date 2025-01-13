echo $@

python pretrain_hx.py \
    agent=$1 \
    domain=$2 \
    seed=$3 \
    use_wandb=true \
    agent.ent_reward_ratio=5 \
    device=$4 \


# agent: aps_sac, diayn_sac, cic_sac


