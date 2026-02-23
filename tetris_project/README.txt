The Tetris project contains the following files:
model_tetris.py - file that contains the parameters to modify for the Tetris environment.
submit_model_tetris.sh - file to submit to slurm on HPC3 in order to run "model_tetris.py" 
tensorboard_video_recorder.py - file to support TensorBoard viewing of the Tetris agent's learning progress. Should not be modified.
tetris_logs - empty folder for model_tetris.py to place its environment logs into, which TensorBoard will use and convert into a progress display. 

To start testing the agent, modify model_tetris.py to adjust any parameters, and then submit the job to slurm using "sbatch submit_model_tetris.sh" on an HPC3 compute node. 
For easier tracking of each individual experiment, change the variable "experiment_name" for each new iteration.
To view the agent's progress, open a TensorBoard server within the folder containing "tetris_logs". 
