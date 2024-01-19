from utils.utils import create_ipynb_file
order = []

for dim in [32, 64, 128]:
    for model in ['3']:
        for density in [0.10, 0.20, 0.30, 0.40, 0.50]:
            string = f'!python Experiment.py --rounds {3} --exper {model} --model {model} '
            string += f'--density {density} '
            string += f'--epochs {150} '
            string += f'--bs {16} --lr {0.001} --decay {0.001} '
            string += f'--dimension {dim} '
            string += f'--experiment 1 '
            string += f'--valid {1}'
            cell = {
                "cell_type" : "code",
                "source" : string,
                "metadata" : {}
            }
            order.append(cell)
create_ipynb_file(order, 'Experiment')
