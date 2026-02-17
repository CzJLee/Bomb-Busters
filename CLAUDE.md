# Claude instructions for Bomb Busters repo

See @README.md for information about Bomb Busters and the rules of the game.

See @Bomb Busters Rulebook.pdf for the official published PDF rulebook of the game. Refer to this for any unclear or ambiguous rules that the README does not cover. The @Bomb Busters FAQ.pdf file contains some additional clarifications. 

## Project Overview

This Bomb Busters project builds a calculator for the game in Python to compute the probability of success of different cut actions. 

### Python game simulator

This repo should create a Python class structure for different game components to allow for easy probability calculations. @bomb_busters.py is the main script. 

### Probability calculations

At this point in time, this project desires to compute the probability of specific events. E.g. "What is the % probability that I successfully dual cut this "2" on this players tile stand." Or, "What guaranteed cut actions do I have?" Or, "What is the highest probability cut action I have?" 

## Architecture

- The python file @bomb_busters.py contains game specific info. 
- The python file @compute_probabilities.py contains python logic to compute probabilities of certain events taking into consideration different sources of information. 

## Environment setup

Use `pyenv virtualenv` to manage the python environment for this repo. The `virtualenv` for this repo is `bomb-busters`. It uses Python 3.14. 