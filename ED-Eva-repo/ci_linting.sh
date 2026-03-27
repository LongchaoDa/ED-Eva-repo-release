#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/src"
output=$(pylint $(git ls-files '*.py'))
pylint_exit_code=$?
error_messages=$(echo "$output" | grep -E 'E[0-9]{4}:|F[0-9]{4}:')
echo "$error_messages"
exit $(($pylint_exit_code & 35)) 

# Exit Code list: https://pylint.readthedocs.io/en/latest/user_guide/usage/run.html#exit-codes