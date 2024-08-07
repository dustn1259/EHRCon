import pandas as pd
import sqlite3
from tqdm import tqdm

def make_specific_db(hadm_id, lab_id, d_items,d_icd_procedures,d_icd_diagnoses,diagnoses,procedures, prescriptions, chart, inputcv, inputmv, micro, output, lab):
    print(f"Creating database for HADM_ID: {hadm_id}...")
    
    def to_lowercase(df):
        for col in df.columns:
            if 'num' in col.lower() or 'amount' in col.lower() or 'rate' in col.lower() or 'hadm_id' in col.lower():
                continue
            elif 'dose_val_rx' in col.lower():
                # Convert to float if possible, else set to 0
                def try_convert_to_float(x):
                    try:
                        return float(x)
                    except ValueError:
                        return 0
                
                df[col] = df[col].apply(try_convert_to_float)
            else:
                df[col] = df[col].astype(str).str.lower()
        return df

    chart_ = to_lowercase(chart[chart['HADM_ID'] == hadm_id].reset_index(drop=True))
    procedures_ = to_lowercase(procedures[procedures['HADM_ID'] == hadm_id].reset_index(drop=True))
    diagnoses_ = to_lowercase(diagnoses[diagnoses['HADM_ID'] == hadm_id].reset_index(drop=True))
    prescriptions_ = to_lowercase(prescriptions[prescriptions['HADM_ID'] == hadm_id].reset_index(drop=True))
    inputcv_ = to_lowercase(inputcv[inputcv['HADM_ID'] == hadm_id].reset_index(drop=True))
    inputmv_ = to_lowercase(inputmv[inputmv['HADM_ID'] == hadm_id].reset_index(drop=True))
    micro_ = to_lowercase(micro[micro['HADM_ID'] == hadm_id].reset_index(drop=True))
    output_ = to_lowercase(output[output['HADM_ID'] == hadm_id].reset_index(drop=True))
    lab_ = to_lowercase(lab[lab['HADM_ID'] == hadm_id].reset_index(drop=True))
    d_items_ = to_lowercase(d_items)
    d_icd_procedures_ = to_lowercase(d_icd_procedures)
    d_icd_diagnoses_ = to_lowercase(d_icd_diagnoses)
    lab_id_ = to_lowercase(lab_id)
    
    conn = sqlite3.connect(f'EHRCon/dataset/database/{hadm_id}.db')
    chart_.to_sql('chartevents', conn, if_exists='replace', index=False)
    prescriptions_.to_sql('prescriptions', conn, if_exists='replace', index=False)
    diagnoses_.to_sql('diagnoses', conn, if_exists='replace', index=False)
    procedures_.to_sql('procedures', conn, if_exists='replace', index=False)
    inputcv_.to_sql('inputevents_cv', conn, if_exists='replace', index=False)
    inputmv_.to_sql('inputevents_mv', conn, if_exists='replace', index=False)
    micro_.to_sql('microbiologyevents', conn, if_exists='replace', index=False)
    output_.to_sql('outputevents', conn, if_exists='replace', index=False)
    lab_.to_sql('labevents', conn, if_exists='replace', index=False)
    lab_id_.to_sql('d_labitems', conn, if_exists='replace', index=False)
    d_items_.to_sql('d_items', conn, if_exists='replace', index=False)
    d_icd_procedures_.to_sql('d_icd_procedures', conn, if_exists='replace', index=False)
    d_icd_diagnoses_.to_sql('d_icd_diagnoses', conn, if_exists='replace', index=False)
    conn.close()


print("start")
data_path = '/path/to/mimic-iii/'
chart = pd.read_csv(f'{data_path}CHARTEVENTS.csv')
inputcv = pd.read_csv(f'{data_path}INPUTEVENTS_CV.csv')
inputmv = pd.read_csv(f'{data_path}INPUTEVENTS_MV.csv')
micro = pd.read_csv(f'{data_path}MICROBIOLOGYEVENTS.csv')
output = pd.read_csv(f'{data_path}OUTPUTEVENTS.csv')
lab = pd.read_csv(f'{data_path}LABEVENTS.csv')
prescriptions = pd.read_csv(f'{data_path}PRESCRIPTIONS.csv')  
d_items = pd.read_csv(f'{data_path}D_ITEMS.csv')  
lab_id = pd.read_csv(f'{data_path}D_LABITEMS.csv') 
d_icd_diagnoses = pd.read_csv(f'{data_path}D_ICD_DIAGNOSES.csv') 
d_icd_procedures = pd.read_csv(f'{data_path}D_ICD_PROCEDURES.csv') 
procedures = pd.read_csv(f'{data_path}PROCEDURES_ICD.csv')  
diagnoses = pd.read_csv(f'{data_path}DIAGNOSES_ICD.csv')  

test_data = pd.read_csv('/ehrcon/csv/file/path')
hadm_ids = test_data['HADM_ID'].tolist()

for hadm_id in tqdm(hadm_ids):
    make_specific_db(hadm_id, lab_id, d_items,d_icd_procedures,d_icd_diagnoses,diagnoses,procedures, prescriptions, chart, inputcv, inputmv, micro, output, lab)