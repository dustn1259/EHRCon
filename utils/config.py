import torch
import yaml

DATA_FILES = {
    'top_items': 'EHRCon/dataset/top_items.pkl',
    'predefined_mapping': 'EHRCon/dataset/predefined_mapping.pickle',
    'abbreviation': 'EHRCon/dataset/final_set_of_abbreviation1.csv',
    'lab_items': 'your/mimic/data/path/D_LABITEMS.csv',
    'd_items': 'your/mimic/data/path/D_ITEMS.csv',
    'prescriptions': 'your/mimic/data/path/PRESCRIPTIONS.csv',
    'icd_procedures': 'your/mimic/data/path/D_ICD_PROCEDURES.csv',
    'icd_diagnoses': 'your/mimic/data/path/D_ICD_DIAGNOSES.csv',
}

DB_PATH = 'EHRCon/dataset/database/'

OPEN_API_KEY = 'your_api_key'
API_BASE = 'your_api_base'
API_MODEL = 'your_model_name'

