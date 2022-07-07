import rlkit.util.hyperparameter as hyp
from launchers.bc_exp_launcher import bc_exp_launcher
path_func = lambda x: '/iris/u/khazatsky/bridge_codebase/datasets/{0}.npy'.format(x)

variant = dict(
	env_name='CarrotPlate-v0',
	horizon=65,
	
	model_kwargs={},
	datapath=path_func('rand_carrot_demos'),

	use_gpu=False,
	log_dir='/iris/u/khazatsky/bridge_codebase/data/',
	)

if __name__ == "__main__":
    search_space = {
        "seed": range(1),
        "demo_size": [500],
        "batch_size": [32, 64, 128,],
        "weight_decay": [0.0, 0.0001, 0.001,],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    exp_id = 0
    for variant in sweeper.iterate_hyperparameters():
    	bc_exp_launcher(variant, run_id=103, exp_id=exp_id)
    	exp_id += 1