rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net_da.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDship2 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 15000
rem python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC2 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 15000 MODEL.WEIGHT ../logSSToSMDship2/model_0015000.pth

rem da_faster_rcnn_R_50_C4_SMD2SS########
python tools/train_net_da.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSship2d DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 15000
python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC2d DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 15000 MODEL.WEIGHT ../logSMDToSSship2d/model_0015000.pth


rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net_da.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDship2 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 15000
python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC101 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000 MODEL.ADV.GA_DIS_LAMBDA 0.1 MODEL.ADV.CA_DIS_LAMBDA 0.01 MODEL.WEIGHT ../logSSToSMDship2/model_0002500.pth

python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC11 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000 MODEL.ADV.GA_DIS_LAMBDA 0.1 MODEL.ADV.CA_DIS_LAMBDA 0.1 MODEL.WEIGHT ../logSSToSMDship2/model_0002500.pth

python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC011 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 20000 MODEL.ADV.GA_DIS_LAMBDA 0.01 MODEL.ADV.CA_DIS_LAMBDA 0.1 MODEL.WEIGHT ../logSSToSMDship2/model_0002500.pth



rem da_faster_rcnn_R_50_C4_SMD2SS########
rem python tools/train_net_da.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSship2 DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 15000
python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC101 DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000 MODEL.ADV.GA_DIS_LAMBDA 0.1 MODEL.ADV.CA_DIS_LAMBDA 0.01 MODEL.WEIGHT ../logSMDToSSship2/model_0002500.pth

python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC11 DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000 MODEL.ADV.GA_DIS_LAMBDA 0.1 MODEL.ADV.CA_DIS_LAMBDA 0.1 MODEL.WEIGHT ../logSMDToSSship2/model_0002500.pth

python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC011 DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 20000 MODEL.ADV.GA_DIS_LAMBDA 0.01 MODEL.ADV.CA_DIS_LAMBDA 0.1 MODEL.WEIGHT ../logSMDToSSship2/model_0002500.pth

