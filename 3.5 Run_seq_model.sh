python run_lstm.py --rounds 5 --density 0.01 --model LSTM --dataset cpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_lstm.py --rounds 5 --density 0.01 --model LSTM --dataset gpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_gru.py --rounds 5 --density 0.01 --model GRU --dataset cpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_gru.py --rounds 5 --density 0.01 --model GRU --dataset gpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_birnn.py --rounds 5 --density 0.01 --model BIRNN --dataset cpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_birnn.py --rounds 5 --density 0.01 --model BIRNN --dataset gpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_arnn.py --rounds 5 --density 0.01 --model ARNN --dataset cpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

python run_arnn.py --rounds 5 --density 0.01 --model ARNN --dataset gpu --epochs 1 --bs 1 --lr 0.0004 --decay 0.0005 --program_test 1 --dimension 32 --experiment 1 --record 1

