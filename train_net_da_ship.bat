rem da_faster_rcnn_R_50_C4_SS2SMD########
rem python tools/train_net_da.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDship2 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 15000
python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC2 DATASETS.TRAIN_SOURCE ('ship_train_SeaShips_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SMD_cocostyle',) DATASETS.TEST ('ship_test_SMD_cocostyle',) SOLVER.MAX_ITER 15000 MODEL.WEIGHT ../logSSToSMDship2/model_0015000.pth





rem da_faster_rcnn_R_50_C4_SMD2SS########
rem python tools/train_net_da.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSship2 DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 15000
python tools/train_net_da.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC2 DATASETS.TRAIN_SOURCE ('ship_train_SMD_cocostyle',) DATASETS.TRAIN_TARGET ('ship_train_SeaShips_cocostyle',) DATASETS.TEST ('ship_test_SeaShips_cocostyle',) SOLVER.MAX_ITER 15000 MODEL.WEIGHT ../logSSToSMDship2/model_0015000.pth


