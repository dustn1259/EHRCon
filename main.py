import yaml
from utils.text_processing import *
from utils.model import *
import argparse
import utils.config as config
import pandas as pd
import pickle
from tqdm import tqdm
import datetime
import sqlite3
from time import time, sleep
from utils.utils import *
import openai

def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_config = yaml.safe_load(file)
    return yaml_config

def main():
    parser = argparse.ArgumentParser(description="Model Loader")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to load')
    parser.add_argument('--config_yaml', type=str, help='Path to the YAML configuration file')
    parser.add_argument('--mode', type=str, choices=['few_shot', 'zero_shot'], default='few_shot', help='Mode to run the model: few_shot or zero_shot (default: few_shot)')
    args = parser.parse_args()
    
    # Load model
    mode = args.mode
    model_name = args.model
    if model_name in ['tulu2', 'mixtral','llama3']:
        model, tokenizer, model_config = load_model(model_name)
    # Load YAML configuration
    yaml_cnfig = load_yaml_config(args.config_yaml)
    py_confoig = {
        "DATA_FILES": config.DATA_FILES,
        "DB_PATH": config.DB_PATH,
        "OPEN_API_KEY": config.OPEN_API_KEY,
        "API_BASE":config.API_BASE,
        "API_MODEL":config.API_MODEL
    }
    note_types = yaml_cnfig['note_types']
    result_path_=yaml_cnfig['result_path']
    OPEN_API_KEY= py_confoig['OPEN_API_KEY']
    API_BASE = py_confoig['API_BASE']
    API_MODEL = py_confoig['API_MODEL']
    df = pd.read_csv(yaml_cnfig['CSV_FILE'])
    with open(py_confoig['DATA_FILES']['top_items'], 'rb') as f:
        top_items = pickle.load(f)
    with open(py_confoig['DATA_FILES']['predefined_mapping'], 'rb') as f:
        predef_map = pickle.load(f)
    sets = pd.read_csv(py_confoig['DATA_FILES']['abbreviation'])
    abb_df = pd.DataFrame(sets, columns=['abbreviation', 'expansion', 'whole_set'])
    
    lab_id = pd.read_csv(py_confoig['DATA_FILES']['lab_items'])
    d_items = pd.read_csv(py_confoig['DATA_FILES']['d_items'])
    prescriptions = pd.read_csv(py_confoig['DATA_FILES']['prescriptions'])
    d_icd_procedures = pd.read_csv(py_confoig['DATA_FILES']['icd_procedures'])
    d_icd_diagnoses = pd.read_csv(py_confoig['DATA_FILES']['icd_diagnoses'])
    
    lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, d_icd_procedures_dict, d_icd_diagnoses_dict, label_keys, d_item_label_keys, prescription_keys, drug_list, d_icd_procedures_label_keys, d_icd_diagnoses_label_keys = create_dicts(lab_id, d_items, prescriptions, d_icd_procedures, d_icd_diagnoses)


    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        split_dict = {}
        text_line_split = row['TEXT'].split('\n') 
        for idx2,line2 in enumerate(text_line_split):
            split_dict[idx2] = line2
        total_dict={}
        hadm_id = row['HADM_ID'] ##333.0
        print("hadm_id:   ",hadm_id)
        hadm_id_string= str(int(hadm_id)) ###333
        text = row['TEXT']
        admission_string = row['ADMITTIME']
        date_object = datetime.datetime.strptime(admission_string, "%Y-%m-%d %H:%M:%S")
        admission = date_object.strftime("%Y-%m-%d")
        charttime = row['CHARTDATE']
        type = row['CATEGORY']
        db_path_=py_confoig['DB_PATH']
        db_path = f'{db_path_}/sample_{hadm_id}.db'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print()
        if model_name in ['tulu2', 'mixtral','llama3']:
            sections,sections_dict_ner=extract_sections(text, hadm_id, 0, model, tokenizer, model_config,yaml_cnfig['prompt_files']['note_segmentation'],model_name)
        else:
            sections,sections_dict_ner=extract_sections_gpt(text, hadm_id, API_MODEL, OPEN_API_KEY, API_BASE,yaml_cnfig['prompt_files']['note_segmentation'],0)

        for sid,section in enumerate(sections):
            total_list=[]
            table_dict={}
            query_output_dict={}
            table_filling={}
            fill_the_table_dict={}
            self_correction_dict_={}
            reformat_value={}
            table_filter=[]
            extracted_list = []
            parsed_input ={}
            print("Number of Section:   ",sid)
            indices_regex=[]
            print()
            ner_prompt_path = yaml_cnfig['prompt_files']['named_entity_recognition']
            if mode == 'zero_shot':
                if note_types == 'discharge':   
                    ner_prompt = open_file(f'{ner_prompt_path}_discharge.txt').replace("<<<CLINICAL_NOTE>>>",sections_dict_ner[sid])
                else:
                    ner_prompt = open_file(f'{ner_prompt_path}.txt').replace("<<<CLINICAL_NOTE>>>",sections_dict_ner[sid])
            else:
                if model_name == 'tulu2':
                    ner_prompt = open_file(f'{ner_prompt_path}_tulu.txt').replace("<<<CLINICAL_NOTE>>>",sections_dict_ner[sid])
                else:    
                    ner_prompt = open_file(f'{ner_prompt_path}.txt').replace("<<<CLINICAL_NOTE>>>",sections_dict_ner[sid])

            if model_name == 'llama3':
                ner_output = open_source_model_inference_llama(ner_prompt, model, tokenizer, model_config)
            elif model_name == 'chatgpt':
                ner_output = chatgpt_completion(ner_prompt, API_MODEL, OPEN_API_KEY, API_BASE)
            else:
                ner_output = open_source_model_inference(ner_prompt, model, tokenizer, model_config)

            if note_types == 'discharge':
                ner_out_=extract_entities_general_dis(ner_output)
            else:
                ner_out_=extract_entities_general_nur_phy(ner_output)

            if ner_out_ == "Step 7 not found in the text." or ner_out_ == "Step 5 not found in the text.":
                continue
            final_list=filter_entities(ner_out_)
            extracted_list = [item['E'] for item in final_list]
            extracted_list=list(set(extracted_list))
            extracted_list=remove_none_from_list(extracted_list)
            print("Named Entity Recognition:   ", extracted_list)
            print()
            for extract in extracted_list:
                word_pattern = create_regex_pattern(extract)
                try:
                    for match in re.finditer(word_pattern, section):
                        start_idx, end_idx = match.start(), match.end() - 1
                        indices_regex.append({extract: [start_idx, end_idx]})  
                except:
                    print("except: ",extract)
                    continue
            temp_dict=remove_overlapping_spans_correctly(indices_regex)
            final_dict_entity=remove_invalid_labels(section,temp_dict)
            transformed_texts_list = transform_text_corrected(section, final_dict_entity)
            for rid,rex in enumerate(final_dict_entity):
                for key_entity, where_idx in rex.items():
                    print("Entity:   ",key_entity)
                    found_line = find_line_containing_word2(section,text, where_idx, key_entity)
                    print()
                    table_iden_path = yaml_cnfig['prompt_files']['table_identification']
                    if mode == 'zero_shot':
                        if note_types == 'discharge':
                            table_identification_prompt = open_file(f'{table_iden_path}/table_identification_discharge.txt').replace('<<<<ENTITY>>>>',key_entity)
                        else:
                            table_identification_prompt = open_file(f'{table_iden_path}/table_identification.txt').replace('<<<<ENTITY>>>>',key_entity)
                    else:
                        table_identification_prompt = open_file(f'{table_iden_path}/table_identification.txt').replace('<<<<ENTITY>>>>',key_entity)
                    if model_name == 'llama3':
                        table_identification_output = open_source_model_inference_llama(table_identification_prompt, model, tokenizer, model_config)
                    elif model_name == 'chatgpt':
                        table_identification_output = chatgpt_completion(table_identification_prompt, API_MODEL, OPEN_API_KEY, API_BASE)
                    else:
                        table_identification_output = open_source_model_inference(table_identification_prompt, model, tokenizer, model_config)
                    table_identification_output=extract_section_ranges(table_identification_output)
                    print("Output of Table Identification:   ", table_identification_output) 
                    print()
                    if "none" not in table_identification_output.lower():
                        transformed_list = transform_to_nested_list(table_identification_output)
                        table_filling[extract]=transformed_list
                        time_filter_path = yaml_cnfig['prompt_files']['time_filtering']
                        if mode == 'zero_shot': 
                            if note_types == 'discharge':
                                time_filter = open_file(f'{time_filter_path}/time_filtering_discharge.txt').replace('<<<CLINICAL_NOTE>>>', transformed_texts_list[rid]).replace('<<<ENTITY>>>',key_entity).replace("<<<CHARTTIME>>>",charttime).replace("<<<ADMISSION>>>",admission)
                            else:
                                time_filter = open_file(f'{time_filter_path}/time_filtering.txt').replace('<<<CLINICAL_NOTE>>>', transformed_texts_list[rid]).replace('<<<ENTITY>>>',key_entity).replace("<<<CHARTTIME>>>",charttime).replace("<<<ADMISSION>>>",admission)
                        else:
                            time_filter = open_file(f'{time_filter_path}/time_filtering.txt').replace('<<<CLINICAL_NOTE>>>', transformed_texts_list[rid]).replace('<<<ENTITY>>>',key_entity).replace("<<<CHARTTIME>>>",charttime).replace("<<<ADMISSION>>>",admission)

                        if model_name == 'llama3':
                            time_output = open_source_model_inference_llama(time_filter, model, tokenizer, model_config)
                        elif model_name == 'chatgpt':
                            time_output = chatgpt_completion(time_filter, API_MODEL, OPEN_API_KEY, API_BASE)
                        else:
                            time_output = open_source_model_inference(time_filter, model, tokenizer, model_config)

                        dict,time_span,time_expression,time_concept=parse_questions_with_return(time_output)
                        question1_value = dict.get('question1', 'Not Found')
                        question3_value = dict.get('question3', 'Not Found')
                        print("Output of Time Filtering: ",question3_value)
                        res_list_by_table = []  
                        for table in transformed_list:
                            print()
                            print("Table to be examined:    ",table)
                            print()
                            extracted_occurence_list_=[]
                            extracted_occurence_list2=[]
                            all_nan = True
                            if table == ['Microbiologyevents', 'd_items']:
                                table = ['Microbiologyevents']
                            for tab in table:
                                tab=tab.lower()
                                if tab == 'procedures_icd':
                                    continue
                                if tab == 'diagnoses_icd':
                                    continue
                                pseudo_table_path = yaml_cnfig['prompt_files']['pseudo_table_creation']
                                fill_table = open_file(f'{pseudo_table_path}/{tab}.txt').replace('<<<<CLINICAL_NOTE>>>>', transformed_texts_list[rid]).replace('<<<ENTITY>>>',key_entity).replace("<<<<Admission>>>>",admission).replace("<<<<Charttime>>>>",charttime).replace("<<<<admission>>>>",admission)
                                if model_name == 'llama3':
                                    fill_table_output = open_source_model_inference_llama(fill_table, model, tokenizer, model_config)
                                elif model_name == 'chatgpt':
                                    fill_table_output = chatgpt_completion(fill_table, API_MODEL, OPEN_API_KEY, API_BASE)
                                else:
                                    fill_table_output = open_source_model_inference(fill_table, model, tokenizer, model_config)
                                extract_output=extract_occurrences(fill_table_output, tab)

                                print(f"Output of Pseudo Table Creation (Table: {tab}):       ",extract_output)
                                extracted_occurence_list_.append(extract_output)
                                extracted_occurence_list2.append([extract_output,table])
                            fill_the_table_dict[key_entity]=extracted_occurence_list2 
                            output_extracted_occurence_list= combine_lists(extracted_occurence_list_)
                            res_list_by_occur = []
                            if table == ['Microbiologyevents']:
                                table = ['Microbiologyevents', 'd_items']
                            for extracted_occurence_list in output_extracted_occurence_list:
                                result_dict={}
                                result_dict[key_entity] = {
                                'label': 'No result',
                                'position_idx': where_idx,
                                'position': found_line,
                                'table_cell' : parsed_input
                            }
                                for occurrence in extracted_occurence_list:
                                    if any(value != 'NaN' for value in occurrence.values()):
                                        all_nan = False
                                        break
                                if not all_nan:
                                    generated_questions=generate_formatted_output(extracted_occurence_list,key_entity)
                                    self_correction_prompt = open_file(yaml_cnfig['prompt_files']['self_correction']).replace("<<<<CLINICAL_NOTE>>>>",transformed_texts_list[rid]).replace("<<<Questions>>>",generated_questions).replace("<<<ENTITY>>>",key_entity).replace("<<<Admission>>>",admission).replace("<<<Charttime>>>",charttime)
                                    if model_name == 'llama3':
                                        self_correction_output = open_source_model_inference_llama(self_correction_prompt, model, tokenizer, model_config)
                                    elif model_name == 'chatgpt':
                                        self_correction_output = chatgpt_completion(self_correction_prompt, API_MODEL, OPEN_API_KEY, API_BASE)
                                    else: 
                                        self_correction_output =  open_source_model_inference(self_correction_prompt, model, tokenizer, model_config)
                                    _self_correction_dict= process_data_with_accurate_parsing(self_correction_output,extracted_occurence_list)  
                                    self_correction_dict_=process_dictionary_case_insensitive(_self_correction_dict)
                                    self_correction_dict=correct_nan_values(self_correction_dict_)

                                    self_correction_dict_[key_entity]=self_correction_dict
                                    
                                    for key, value in self_correction_dict.items():
                                        if value == 'NaN.':
                                            self_correction_dict[key] = 'NaN'
                                    print()
                                    print("Output of Self Correction:    ",self_correction_dict)
                                    print()

                                    all_nan_ = all(value == 'NaN' for value in self_correction_dict.values())
                                    if not all_nan_:
                                        formatted = format_input_for_tables(self_correction_dict,table)
                                        formatted_string = "\n".join(formatted)
                                        format_list=extract_name_item_list(formatted_string)
                                        table_lower = [item.lower() for item in table]
                                        table_candidate = ["chartevents","inputevents_cv","inputevents_mv", "labevents", "microbiologyevents","outputevents","prescriptions","diagnoses_icd","procedures_icd"]
                                        overlap = [item for item in table_candidate if item in table_lower]
                                        overlap=overlap[0]
                                        reformat_prompt_path = yaml_cnfig['prompt_files']['value_reformatting']
                                        reformat_prompt = open_file(f'{reformat_prompt_path}/{overlap}.txt').replace("<<<<GIVEN_DATA>>>>",formatted_string).replace("<<<<Admission>>>>",admission).replace("<<<<Charttime>>>>",charttime) 
                                        if model_name == 'llama3':
                                            reformat_output = open_source_model_inference_llama(reformat_prompt, model, tokenizer, model_config)
                                        elif model_name == 'chatgpt':
                                            reformat_output = chatgpt_completion(reformat_prompt, API_MODEL, OPEN_API_KEY, API_BASE)
                                        else: 
                                            reformat_output = open_source_model_inference(reformat_prompt, model, tokenizer, model_config)
                                        reformat_output=reformat_output.replace(": ","=")
                                        try:
                                            reformat_samples=extract_matching_items_ignore_case(reformat_output,format_list)
                                            reformat_samples=reformat_samples.replace("[**","").replace("**]","")
                                            reformat_value[key_entity]=reformat_samples
                                        except:
                                            
                                            continue
                                        reformat_samples = filter_lines(reformat_samples)
                                        parsed_input=parse_input(reformat_samples)
                                        print("Output of Value Reformatting:  ",parsed_input)
                                        print()
                                        if note_types == 'discharge':
                                            sql_number= sql_selection(question3_value,self_correction_dict)
                                        else:
                                            sql_number= sql_selection_not_dis(question3_value,self_correction_dict)
                                        print("Query Type: ",sql_number)
                                        overlap_=overlap.capitalize()
                                        sql_generate_path = yaml_cnfig['prompt_files']['sql_generate']
                                        if sql_number == 'SQL_0':  ##no time
                                            if overlap_.lower() == 'procedures_icd':
                                                reformat_samples=remove_specific_lines(reformat_samples)
                                                reformat_samples=f"d_icd_procedures.LONG_TITLE={key_entity}\n d_icd_procedures.SHORT_TITLE={key_entity} "
                                            elif overlap_.lower() == 'diagnoses_icd':
                                                reformat_samples=remove_specific_lines(reformat_samples)
                                                reformat_samples=f"d_icd_diagnoses.LONG_TITLE={key_entity}\n d_icd_diagnoses.SHORT_TITLE={key_entity} "
                                            reformat_samples=remove_specific_lines(reformat_samples)
                                            keys = find_keys_for_tables(table)
                                            if len(keys)>1:
                                                join_keys=" and ".join(keys)
                                            elif len(keys)==1:
                                                join_keys = keys[0]
                                            else:
                                                join_keys = ''
                                            time_value_=find_times_for_tables(table)
                                            time_value=str(time_value_)
                                            if 'Prescriptions' in table or 'prescriptions' in table:
                                                no_time_prompt=open_file(f'{sql_generate_path}/no_time/{overlap}.txt').replace("<<<TIME_VALUES>>>",time_value).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples).replace("3. The '<<<TABLE_1>>>' and '<<<TABLE_2>>>' tables need to be joined using the '<<<JOIN_KEY>>>' as the key for the join operation.","")                                                              
                                            elif 'procedures_icd' in table or 'procedures_icd' in table: 
                                                no_time_prompt=open_file(f'{sql_generate_path}/no_time/procedures_icd.txt').replace("<<<TIME_VALUES>>>",time_value).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples)                                                           
                                            elif 'diagnoses_icd' in table or 'Diagnoses_icd' in table: 
                                                no_time_prompt=open_file(f'{sql_generate_path}/no_time/diagnoses_icd.txt').replace("<<<TIME_VALUES>>>",time_value).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples)
                                            else:
                                                no_time_prompt=open_file(f'/home/yskwon/Notable_Check/mimic/openmodels/main/few_shot/discharge/prompt/sql_generate/no_time/{overlap}.txt').replace("<<<TABLE_1>>>",table[0]).replace("<<<TABLE_2>>>",table[1]).replace("<<<JOIN_KEY>>>",join_keys).replace("<<<TIME_VALUES>>>",time_value).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples)                      
                                            if model_name == 'llama3':
                                                no_time_output = open_source_model_inference_llama(no_time_prompt, model, tokenizer, model_config)
                                            elif model_name == 'chatgpt':
                                                no_time_output = chatgpt_completion(no_time_prompt, API_MODEL, OPEN_API_KEY, API_BASE)
                                            else:
                                                no_time_output = open_source_model_inference(no_time_prompt, model, tokenizer, model_config)
                                            no_time_output=extract_section_ranges_sql(no_time_output)
                                            no_time_output=no_time_output.replace("[","").replace("]","").replace('"Q"',"'Q'")
                                            try:
                                                query_output_dict[key_entity]=clean_and_eval_query(no_time_output)['Q']
                                                print("query:     ", clean_and_eval_query(no_time_output)['Q'])
                                                generated_sqls = total_checker(top_items,predef_map, key_entity, clean_and_eval_query(no_time_output)['Q'], lab_id, drug_list, d_items, lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, d_icd_procedures, d_icd_diagnoses, n=2, fir_thr=0.5, sec_thr=0.7)
                                                result_dict[key_entity] = {
                                                    'label': 'No result',
                                                    'position_idx': where_idx,
                                                    'position': found_line,
                                                    'table_cell' : parsed_input
                                                }
                                                for query in generated_sqls:
                                                    if 'error' == query[1]:
                                                        break
                                                    try:
                                                        cursor.execute(query[1].replace("%y","%Y").replace(":%m",":%M").replace("%h","%H").replace(f"%s",f"%S"))
                                                        results = cursor.fetchall()
                                                        result_dict[key_entity]['label'] = 'False'

                                                    except:
                                                        continue
 
                                                    if len(results) > 0:
                                                        result_dict[key_entity]['label'] = 'True'

                                                        break
                                                    else:
                                                        continue
                                            except:
                                                print("no_sql_generate")
                                                continue
                                        elif sql_number =='SQL_3':
                                            continue    
                                        
                                        elif sql_number == 'SQL_1': 
                                            keys = find_keys_for_tables(table)
                                            if len(keys)>1:
                                                join_keys=" and ".join(keys)
                                            elif len(keys)==1:
                                                join_keys = keys[0]
                                            else:
                                                join_keys = ''
                                            if 'Prescriptions' in table or 'prescriptions' in table:
                                                exact_time_prompt=open_file(f'{sql_generate_path}/exact_time/{overlap}.txt').replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples).replace("2. The '<<<TABLE_1>>>' and '<<<TABLE_2>>>' tables need to be joined using the '<<<JOIN_KEY>>>' as the key for the join operation.",'')                               
                                            else:
                                                exact_time_prompt=open_file(f'{sql_generate_path}/exact_time/{overlap}.txt').replace("<<<TABLE_1>>>",table[0]).replace("<<<TABLE_2>>>",table[1]).replace("<<<JOIN_KEY>>>",join_keys).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples)                            
                                            if model_name == 'llama3':
                                                exact_time_output = open_source_model_inference_llama(exact_time_prompt, model, tokenizer, config)
                                            elif model_name == 'chatgpt':
                                                exact_time_output = chatgpt_completion(exact_time_prompt,  API_MODEL, OPEN_API_KEY, API_BASE)
                                            else:
                                                exact_time_output = open_source_model_inference(exact_time_prompt, model, tokenizer, config) 
                                            exact_time_output=extract_section_ranges_sql(exact_time_output)
                                            exact_time_output=exact_time_output.replace("[","").replace("]","").replace('"Q"',"'Q'")
                                            try:
                                                query_output_dict[key_entity]=clean_and_eval_query(exact_time_output)['Q']
                                                print("query:     ", clean_and_eval_query(exact_time_output)['Q'])
                                                generated_sqls =total_checker(top_items,predef_map, key_entity, clean_and_eval_query(exact_time_output)['Q'], lab_id, drug_list, d_items, lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, d_icd_procedures, d_icd_diagnoses, n=2, fir_thr=0.5, sec_thr=0.7)
                                                result_dict[key_entity] = {
                                                    'label': 'No result',
                                                    'position_idx': where_idx,
                                                    'position': found_line,
                                                    'table_cell' : parsed_input
                                                }
                                                for query in generated_sqls:
                                                    if 'error' == query[1]:
                                                        print("inconsistency: ", query)
                                                        break
                                                    try:
                                                        cursor.execute(query[1].replace("%y","%Y").replace(":%m",":%M").replace("%h","%H").replace(f"%s",f"%S"))
                                                        results = cursor.fetchall()
                                                        result_dict[key_entity]['label'] = 'False'

                                                    except:
                                                        continue
                                                    if len(results) > 0:
                                                        result_dict[key_entity]['label'] = 'True'

                                                        break
                                                    else:
                                                        print("inconsistency: ", query[0])
                                            except:
                                                print("no_sql_generate")
                                                continue
                                        else:
                                            keys = find_keys_for_tables(table)
                                            print("find_table_key?:", keys)
                                            print("check keys: ", keys)
                                            if len(keys)>1:
                                                join_keys=" and ".join(keys)
                                            elif len(keys)==1:
                                                join_keys = keys[0]
                                            else:
                                                join_keys = ''
                                            if ('startdate' in reformat_samples.lower() and 'enddate' in reformat_samples.lower()):
                                                time_key = "Prescriptions.STARTDATE and Prescriptions.ENDDATE"
                                            elif ('starttime' in reformat_samples.lower() and 'endtime' in reformat_samples.lower()): 
                                                time_key = "Inputevents_mv.STARTTIME and Inputevents_mv.ENDTIME"
                                            elif ('startdate' in reformat_samples.lower() and not 'enddate' in reformat_samples.lower()):
                                                time_key = "Prescriptions.STARTDATE"
                                            elif ('enddate' in reformat_samples.lower() and not 'startdate' in reformat_samples.lower()):
                                                time_key = "Prescriptions.ENDDATE"
                                            elif ('starttime' in reformat_samples.lower() and not 'endtime' in reformat_samples.lower()):
                                                time_key = "Inputevents_mv.STARTTIME"
                                            elif ('endtime' in reformat_samples.lower() and not 'starttime' in reformat_samples.lower()):
                                                time_key = "Inputevents_mv.ENDTIME"
                                            elif ('charttime' in reformat_samples.lower()):
                                                time_key = f"{overlap_}.CHARTTIME"
                                            if 'Prescriptions' in table or 'prescriptions' in table:
                                                specific_time_prompt=open_file(f'{sql_generate_path}/specific_time/{overlap}.txt').replace("<<<TIME_VALUES>>>",time_key).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples).replace("3. The '<<<TABLE_1>>>' and '<<<TABLE_2>>>' tables need to be joined using the '<<<JOIN_KEY>>>' as the key for the join operation.","")                                     
                                            else:
                                                specific_time_prompt=open_file(f'{sql_generate_path}/specific_time/{overlap}.txt').replace("<<<TABLE_1>>>",table[0]).replace("<<<TABLE_2>>>",table[1]).replace("<<<JOIN_KEY>>>",join_keys).replace("<<<TIME_VALUES>>>",time_key).replace("<<<ADMISSION>>>",admission).replace("<<<CHARTTIME>>>",charttime).replace("<<<HADM_ID_TABLE>>>",overlap_).replace("<<<HAMD_ID>>>",hadm_id_string).replace("<<<CONDITION_VALUE>>>",reformat_samples)        
                                            if model_name == 'llama3':
                                                specific_time_outputt = open_source_model_inference_llama(specific_time_prompt, model, tokenizer, config)
                                            elif model_name == 'chatgpt':
                                                specific_time_outputt = chatgpt_completion(specific_time_prompt, API_MODEL, OPEN_API_KEY, API_BASE)
                                            else:
                                                specific_time_outputt = open_source_model_inference(specific_time_prompt, model, tokenizer, config)
                                            specific_time_outputt=extract_section_ranges_sql(specific_time_outputt)
                                            specific_time_outputt=specific_time_outputt.replace("[","").replace("]","").replace('"Q"',"'Q'")
                                            #print("query_made:", specific_time_outputt)
                                            try:
                                                query_output_dict[key_entity]=clean_and_eval_query(specific_time_outputt)['Q']
                                                
                                                
                                                print("query:     ", clean_and_eval_query(specific_time_outputt)['Q'])
                                                generated_sqls = total_checker(top_items,predef_map, key_entity, clean_and_eval_query(specific_time_outputt)['Q'], lab_id, drug_list, d_items, lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, d_icd_procedures, d_icd_diagnoses, n=2, fir_thr=0.5, sec_thr=0.7)
                                                result_dict[key_entity] = {
                                                    'label': 'No result',
                                                    'position_idx': where_idx,
                                                    'position': found_line,
                                                    'table_cell' : parsed_input
                                                }
                                                for query in generated_sqls:
                                                    if 'error' == query[1]:
                                                        print("inconsistency: ", query)
                                                        break
                                                    try:
                                                        cursor.execute(query[1].replace("%y","%Y").replace(":%m",":%M").replace("%h","%H").replace(f"%s",f"%S"))
                                                        results = cursor.fetchall()
                                                        #result_dict[key_entity] = {'label':'False'}
                                                        result_dict[key_entity]['label'] = 'False'

                                                    except:
                                                        continue
                                                    if len(results) > 0:
                                                        result_dict[key_entity]['label'] = 'True'

                                                        break
                                                    else:
                                                        print("inconsistency: ", query[0])
                                            except:
                                                continue
                                        
                                    else:
                                        print("All values are NaN, moving to the next table.")
                                        continue
                                #print("result_dict: ",result_dict)
                                res_list_by_occur.append(result_dict)   
                                print("Output: ",res_list_by_occur) 

                            res_list_by_table.append(res_list_by_occur)
                            #print("res_list_by_table: ",res_list_by_table)
                        
                        temp_result = {}
                        temp_result[key_entity] = {
                                                    'label': 'No result',
                                                    'position_idx': where_idx,
                                                    'position': found_line,
                                                    'table_cell' : parsed_input
                                                }
                        
                        return_list_=processing_before_store_data(res_list_by_table)
                        print("Final Output: ", return_list_)
                        print("*********************")
                        print()
                        if len(return_list_) > 0:
                            for temp_result in return_list_:
                                total_list.append(temp_result)

                total_dict[sid]=total_list
                
                with open(f'{result_path_}/results_{hadm_id}.pkl', 'wb') as handle:
                    pickle.dump(total_dict, handle)                
if __name__ == "__main__":
    main()