python run_machinelearning.py --rounds 5 --density 0.01 --dataset cpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.02 --dataset cpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.03 --dataset cpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.04 --dataset cpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.05 --dataset cpu --experiment 1 --record 1

python run_machinelearning.py --rounds 5 --density 0.01 --dataset gpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.02 --dataset gpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.03 --dataset gpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.04 --dataset gpu --experiment 1 --record 1
python run_machinelearning.py --rounds 5 --density 0.05 --dataset gpu --experiment 1 --record 1

python run_mlp.py --rounds 5 --density 0.01 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.02 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.03 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.04 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.05 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
   
python run_mlp.py --rounds 5 --density 0.01 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.02 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.03 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.04 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_mlp.py --rounds 5 --density 0.05 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
   
python run_bpr_nas.py --rounds 5 --density 0.01 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.02 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.03 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.04 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.05 --dataset cpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_bpr_nas.py --rounds 5 --density 0.01 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.02 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.03 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.04 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1
python run_bpr_nas.py --rounds 5 --density 0.05 --dataset gpu --epochs 300 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

