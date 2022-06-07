rem python tools/test_net.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSMDToSSshipC DATASETS.TEST ('ship_test_SeaShips_cocostyle',)
python tools/test_net.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml --checkpoint model_0020000.pth OUTPUT_DIR ../logSSToSMDshipC DATASETS.TEST ('ship_test_SMD_cocostyle',)
rem --flagVisual True
python tools/test_net.py --config-file configs/da_ship/da_ga_ca_ship_R_50_FPN_4x.yaml --flagVisual True --checkpoint model_0020000.pth OUTPUT_DIR ../logSSToSMDshipC101 DATASETS.TEST ('ship_test_SMD_cocostyle',)







