oarsub -t deploy -l nodes=4,walltime=4:00 -p chiclet "/home/agennuso/aware-deployment-grid5k/run_rl_experiments.sh & sleep 10000000" 



oarsub -t deploy -l nodes=4,walltime=4:00 -p chirop -q testing "/home/agennuso/aware-deployment-grid5k/run_rl_experiments.sh & sleep 10000000"


oarsub -t deploy -l nodes=6,walltime=14:00 -p chifflot "/home/agennuso/aware-deployment-grid5k/run_times_experiments.sh & sleep 10000000"


oarsub -t deploy -l nodes=8,walltime=49:30 -p chiclet "/home/agennuso/aware-deployment-grid5k/run_rl_experiments.sh & sleep 100000000" -r "2024-08-17 07:00:00"