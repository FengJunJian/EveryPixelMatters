python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0002500.pth
python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0002500.pth

python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0002500.pth
python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0002500.pth


python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth
python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSshipC DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth


python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSSToSMDship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth
python tools/featureExtractor.py --config-file configs/da_ship/da_ga_ship_R_50_FPN_4x.yaml OUTPUT_DIR ../logSMDToSSship DATASETS.TEST ('ship_test_SMD_cocostyle','ship_test_SeaShips_cocostyle') MODEL.WEIGHT model_0020000.pth

