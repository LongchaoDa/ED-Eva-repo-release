#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"

pytest tests/  \
		--cov-branch \
		--cov-report=term \
		--cov-report=html \
		--cov-report=xml \
		--cov=src/package_name \
		"$@" 