CONTROL=$(head -n 1 $OAR_NODEFILE)
ssh root@$CONTROL "pip install -r gymnasium pettingzoo stable-baselines3 torch numpy matplotlib "