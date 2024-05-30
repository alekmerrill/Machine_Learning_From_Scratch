Programming Assignment 4 (Car.py, Project_5.py, Project5_testing.py, Project_5_Video.py)
@author - Aleksander Merrill

What is this?
-------------

This set of python programs is meant to satisfy the problems present in Programming Assignment 5 and build understanding of reinforcement machine learning through the
   testing of a car on different tracks.
Car.py is a custom class designed to emulate a car with very basic kinematics, turning, accelerating with a chance to fail, and preservation of velocity through angular
   momentum.
The Project_5.py is a set of functions that form the algorithms and support functions. These include Value Iteration, Q-Learning, and SARSA.
Project_5_testing.py is the actual results of the tests, generation of graphs, and use of functions. It imports these functions directly from Project_5.py and
   Car.py.
Project_5_Video.py was only used for the purpose of generating the 3-7 minute video required.

How to Use The Programs
-----------------------

In the correct directory, ensure that the three .txt files are present.

While in this directory within a command prompt run:

python Car.py

python Project_5.py

The above steps are necessary to run first in the above order so as to prevent crashes from lack of prerequisite file for imported functions.

Next run:

python Project_5_testing.py
	