# AMSES-MSF-FSL
Description
This repository includes codes and data for AMSES-MSF-FSL. AMSES-MSF-FSL is An Autonomous Predictive Model Construction Framework

Code
To run AMSES-MSF-FSL on Sock-shop, run the following command:

python test_sock_shop.py --mode --data_path <data_path> --dataset

To run AMSES-MSF-FSL on Train-Ticket, run the following command:

python test_train_ticket.py --mode --data_path <data_path> --dataset

where can be either of 'pretrain', 'train', 'test', 'pre_rca', 'rca'. Among them, 'pretrain' means executing Auto-selection of AMSES, 'train' represents running Fault Detection Model Training of AMSES. 'test' means executing Model Auto-Stacking and Real-time Fault Type Identification, 'pre_rca' means running Root Cause Localization Model Auto-Selection, and 'rca' represents locating fault microservices.
