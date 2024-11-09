build:
	heroku container:push web -a pt-neuralnet-pricing
release:
	heroku container:release web -a pt-neuralnet-pricing
restart:
	heroku restart -a pt-neuralnet-pricing
stop:
	heroku stop dyno -a pt-neuralnet-pricing
