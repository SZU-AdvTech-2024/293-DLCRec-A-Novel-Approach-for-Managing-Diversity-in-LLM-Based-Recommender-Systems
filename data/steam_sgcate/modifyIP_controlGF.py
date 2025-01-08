import json
import re
from tqdm import tqdm

# modify the instrucition to make use of the control of GP

GF_paths = [
    # "./steam_sgcate_result/BERT_GF_vlGF_ep25_1e-3_controlGP_ep25_1_-1_5e-4_0.05_2GF1000gred.json",
    # "./steam_sgcate_result/BERT_GF_vlGF_ep25_1e-3_controlGP_ep25_1_-1_5e-4_0.05_5GF1000gred.json",
    # "./steam_sgcate_result/BERT_GF_vlGF_ep25_1e-3_controlGP_ep25_1_-1_5e-4_0.05_8GF1000gred.json",
    # "./steam_sgcate_result/BERT_GFGFd_vlGF_ep25_1e-3_controlGP_ep25_1_-1_5e-4_0.05_2GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_1GP1000gred_1GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_2GP1000gred_2GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_3GP1000gred_3GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_4GP1000gred_4GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_5GP1000gred_5GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_6GP1000gred_6GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_7GP1000gred_7GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_8GP1000gred_8GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_9GP1000gred_9GF1000gred.json",
    "./steam_sgcate_result/BERT_GF_nd_vlGF_nd_ep25_1e-3_controlBERT_GP_vlGP_ep25_1e-3_10GP1000gred_10GF1000gred.json",
    # "./steam_sgcate_result/BERT_GFGFd_vlGF_ep25_1e-3_controlGP_ep25_1_-1_5e-4_0.05_8GF1000gred.json",

]

IP_test_data_path = './data/steam_sgcate/test_1000_BERT_IP.json'
IP_test_data_path_controlGF = './data//steam_sgcate/test_1000_BERT_IP_controlGF.json'

GF_predictions = {}
IP_movie_pattern = '\"_\"'

for p in GF_paths:
    with open(p, 'r') as f:
        test_data_GF = json.load(f)
        predict = [_['predict'] for _ in test_data_GF]
        GF_predictions[p] = predict

with open(IP_test_data_path, 'r') as f:
    test_data = json.load(f) 
    for i, test in enumerate(test_data):
        test_data[i]['input_controlGF'] = {}
        for p in GF_paths:
            replaced_string = re.sub(IP_movie_pattern, lambda match: '\"?\"', GF_predictions[p][i][0])
            key_p = p.split('/')[-1][:-5].split("GF1000gred")[0]
            test_data[i]['input_controlGF'][key_p] = replaced_string
            
with open(IP_test_data_path_controlGF, 'w') as f:
    json.dump(test_data, f, indent=4)
