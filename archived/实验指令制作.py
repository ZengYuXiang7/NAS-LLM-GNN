from utils.utils import create_sh_file
order = []
for dim in [10, 20, 30, 40, 50]:
    for dataset in ['cpu', 'gpu']:
        for density in [0.05]:
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

create_sh_file(order, 'Run_hyper')


