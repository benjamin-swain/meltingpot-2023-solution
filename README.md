# Melting Pot 2023 Competition Submission - Team Marlshmallows

This repository contains my submission for the Melting Pot 2023 Competition. Detailed information about my solution strategy can be found in the document [Team Marlshmallows Melting Pot 2023 Solution.pdf](Team%20Marlshmallows%20Melting%20Pot%202023%20Solution.pdf).

## Setup Guide

Follow these steps to set up the environment:

### Creating a Conda Environment

1. Create a Conda environment with Python 3.10:
`conda create -n marlshmallows_env python=3.10.12`
2. Activate the newly created environment:
`conda activate marlshmallows_env`

### Installing Necessary Packages

1. Navigate to the project directory
`cd path/to/meltingpot-2023-solution`
2. Make the installation script executable:
`chmod +x install.sh`
3. Run the installation script
`./install.sh`

## Evaluate Policies

To evaluate the policies:

1. Open `evaluate.py` and adjust the user settings at the top of the file.
2. Run the evaluation script:
`python evaluate.py`
