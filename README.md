# gym-pybullet-drones project RO47005

## Description
- Course code: RO47005
- Project group: 2

This repository contains the scripts that were used to obtain the results for the groupproject of the Planning and Decision Making course (RO47005). The simulation environment is premade in PyBullet and from the original `gym-pybullet-drones` repository. 

## Prerequisites/installation 

### Gym-pybullet-drones

Programs that need to be installed before being able to run the scripts are the following:

- Gym-pybullet-drones environment
- Acados solver

The installation instructions of the `gym-pybullet-drones` environment are found in the README of its github page which can be accessed using the following link:
//github.com/utiasDSL/gym-pybullet-drones. After succesful installation, the `drones` python environment should be available and activated.

### Acados

The Acados solver that is used can be installed by following the instructions found at the official Acados website. The website can be accessed with the following url: https://docs.acados.org/installation/ 
The Acados solver should be installed in the `drones` python environment.

## Use

Firstly, clone the github repository by running the following command in a newly created directory:

```
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
```
Then, activate the `drones` environment by running the following command:

```
conda activate drones
```

Then, navigate to `YourDir/gym-pybullet-drones/gym-pybullet-drones/scripts`. The 

To recreate the results obtained in the project the following command should be used.

```
python3 acados_pybullet_pid.py 
```

The script above uses two custom added scripts. These are the `quadrotor_dynamic_model_test` for the dynamic model and `acados_main` where the functions used by Acados are defined.







 