cd experiments_group
export GROUP_FILE=$(pwd)/ex4.txt
cd ..
kopf run hooks_scaling_times_experiments.py --dev --standalone -n default