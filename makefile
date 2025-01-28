build:
	heroku container:push web -a pt-neuralnet-pricing
release:
	heroku container:release web -a pt-neuralnet-pricing
restart:
	heroku restart -a pt-neuralnet-pricing
stop:
	heroku stop --dyno-name web -a pt-neuralnet-pricing
exec_tiny:
	python3 scripts/exec_tinymodel.py
run_bs_calc:
	python3 scripts/run_bs_calc.py
run_heston_calc:
	python3 scripts/calc_heston.py
run_heston2_calc:
	python3 scripts/calc_heston2.py
run_gradient_calc:
	python3 scripts/run_gradient_calc.py
run_tiny:
	python3 scripts/run_tinymodel.py
run_simple:
	python3 scripts/simple_nn.py
run_tensor_demo:
	python3 scripts/tensor_demo.py
run_simple_tensor:
	python3 scripts/run_simple_tensor.py
run_visual:
	python3 scripts/viz.py
