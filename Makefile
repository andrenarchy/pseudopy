
default:
	@echo "\"make upload\"?"

upload: setup.py
	python setup.py sdist upload --sign
