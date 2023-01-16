This folder contains all python code needed to recreate the real-data simulation

Files should be run in the following order

Initial_Filtering
PMI_calc
PMI_est
Calc_EM
Calc_DD
Calc_W2V
Embed_Matrices
Calc_LSTM
Avg_Results

All files except for Initial filtering were run
using separate bash scripts with a job number of 1-X
these job numbers determined the cross valalidation sample used
and the number of negative samples used in each model

All bash scripts have been compiled in order in the Bash.txt document

Also note these files are used by workflow.sh in the parent folder to recreate the simulation.