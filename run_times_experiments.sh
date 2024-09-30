cd $HOME/aware-deployment-grid5k
./deploy.sh
CONTROL=$(head -n 1 $OAR_NODEFILE)
NUM_NODES=$(uniq $OAR_NODEFILE | wc -l)
echo $NUM_NODES
COMMAND="(cd /home/agennuso/aware-deployment-grid5k/rl_operator/times_experiments/; ./deploy_experiment.sh $NUM_NODES)"
ssh root@$CONTROL 'tmux kill-server'
ssh root@$CONTROL "tmux new-session -d -s times_session \"$COMMAND\""
echo "Times experiment started in a tmux session"


#tmux new-session -d -s times_session "cd /home/agennuso/aware-deployment-grid5k/rl_operator/times_experiments/; ./deploy_experiment.sh $NUM_NODES"
#tmux new-session -d -s times_session "cd /home/agennuso/aware-deployment-grid5k/rl_operator/times_experiments/; ./deploy_experiment.sh 3"
#tmux new-session -d -s times_session "cd /home/agennuso/aware-deployment-grid5k/rl_operator/times_experiments/; ./deploy_experiment.sh 4"