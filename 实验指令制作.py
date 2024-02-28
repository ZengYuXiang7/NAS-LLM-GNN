from utils.utils import create_sh_file
order = []
for dim in [16, 32, 128, 256]:
    for dataset in ['cpu', 'gpu']:
        for density in [0.01, 0.02, 0.03, 0.04, 0.05]:
            string = f'python run_llm_gcn.py --rounds {5} '
            string += f'--density {density} '
            string += f'--dataset {dataset} '
            string += f'--epochs {300} '
            string += f'--bs {1} --lr {4e-4} --decay {5e-4} '
            string += f'--program_test {1} '
            string += f'--dimension {dim} '
            string += f'--experiment {1} --record {1}'
            order.append(string)
        order.append('   ')

create_sh_file(order, 'Experiment-hyper-dim')
