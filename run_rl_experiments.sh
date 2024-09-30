cd $HOME/aware-deployment-grid5k
./deploy.sh
CONTROL=$(head -n 1 $OAR_NODEFILE)
ssh root@$CONTROL 'tmux new-session -d -s rl_session "(cd /home/agennuso/aware-deployment-grid5k/rl_operator/rl/Cluster-Environment/cluster-environment/env; python main_ppo_new.py)"'
echo "RL experiment started in a tmux session"



# tmux new-session -d -s rl_session "(cd /home/agennuso/Boh/rl_operator/rl/Cluster-Environment/cluster-environment/env; python main_ppo_new.py)"