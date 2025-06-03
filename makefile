build:
	heroku container:push web -a pt-neuralnet-pricing
release:
	heroku container:release web -a pt-neuralnet-pricing
restart:
	heroku restart -a pt-neuralnet-pricing
stop:
	heroku stop --dyno-name web -a pt-neuralnet-pricing
exec_tiny:
	python3 scripts/utils/exec_tinymodel.py
run_bs_calc:
	python3 scripts/blackscholes/run_bs_calc.py
run_heston_calc:
	python3 scripts/heston/calc_heston.py
run_heston2_calc:
	python3 scripts/heston/calc_heston2.py
run_gradient_calc:
	python3 scripts/utils/run_gradient_calc.py
run_tiny:
	python3 scripts/utils/run_tinymodel.py
run_simple:
	python3 scripts/utils/simple_nn.py
run_tensor_demo:
	python3 scripts/utils/tensor_demo.py
run_simple_tensor:
	python3 scripts/utils/run_simple_tensor.py
run_visual:
	python3 scripts/visualization/viz.py
