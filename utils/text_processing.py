import re
import ast
from utils.utils import *
from utils.model import open_source_model_inference, open_source_model_inference_llama,chatgpt_completion
import string
from strsimpy.cosine import Cosine
import pandas as pd 
from transformers import GPT2Tokenizer
from time import time, sleep
import openai

def transform_to_nested_list(input_string):
    
    valid_pairs = {
        "Chartevents": ["d_items", "D_items"],
        "chartevents": ["d_items", "D_items"],
        "Diagnoses_icd": ["d_icd_diagnoses", "D_icd_diagnoses"],
        "diagnoses_icd": ["d_icd_diagnoses", "D_icd_diagnoses"],
        "Procedures_icd": ["d_icd_procedures", "D_icd_procedures"],
        "procedures_icd": ["d_icd_procedures", "D_icd_procedures"],
        "Outputevents": ["d_items", "D_items"],
        "outputevents": ["d_items", "D_items"],
        "Microbiologyevents": ["d_items", "D_items"],
        "microbiologyevents": ["d_items", "D_items"],
        "Inputevents_mv": ["d_items", "D_items"],
        "inputevents_mv": ["d_items", "D_items"],
        "Inputevents_cv": ["d_items", "D_items"],
        "inputevents_cv": ["d_items", "D_items"],
        "Labevents": ["d_labitems", "D_labitems"],
        "labevents": ["d_labitems", "D_labitems"],
        "Prescriptions": [None],
        "prescriptions": [None]
    }
    
    nested_list = []
    added_pairs = set()
    
    matches = re.findall(r'{([^}]*)}', input_string, re.IGNORECASE)
    
    for match in matches:
        items = [item.strip() for item in match.split(',') if item.strip().lower() != 'none']    
        if items and items[0] in valid_pairs:
            valid_values = valid_pairs[items[0]]
            if tuple(items) not in added_pairs:
                if len(items) == 2 and items[1] in valid_values:
                    nested_list.append(items)
                    added_pairs.add(tuple(items))
                elif len(items) == 1 and None in valid_values:
                    nested_list.append([items[0]])
                    added_pairs.add((items[0],))

    return nested_list

def filter_lines(input_string):
    candidate_list = ['VALUENUM', 'DOSE_VAL_RX', 'AMOUNT', 'RATE']
    lines = input_string.split('\n')
    updated_lines = []
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            key_end = key.split('.')[-1]
            if key_end in candidate_list and '-' in value:
                continue  
        updated_lines.append(line)
    return '\n'.join(updated_lines)

def extract_section_info(section_output):
    sections = {}
    pattern = r'(?i)(section\s*\d+):\s*\(?\s*(\d+)-(\d+)\s*\)?'
    matches = re.findall(pattern, section_output)
    
    output = {section.lower().replace(" ", ""): (int(start), int(end)) for section, start, end in matches}
    
    if 'section3' not in output and 'section2' in output:
        output['section3'] = (output['section2'][1], output['section2'][1])
    
    return output

def extract_section_ranges(text):
    start_idx = text.find('[')
    end_idx = text.find(']') 
    sections_str = text[start_idx:end_idx+1]
    return sections_str

def extract_section_ranges_sql(text):
    start_idx = text.find('[')
    end_idx = text.find(']')
    
    sections_str = text[start_idx:end_idx+1]
    if sections_str == '':
        sections_str = text
    return sections_str

def extract_sections(text, hadm_id, start_line, model, tokenizer, config,seg_path,model_name):
    count_while= 0
    line_splitted = text.split('\n')
    line_numbers = list(range(len(line_splitted)))
    original_tokens = tokenizer.tokenize(text)
    sections_dict = []
    sections_dict_ner = []
    while len(original_tokens) > 1000: 
        numbered_lines = [f"{i}: {line}" for i, line in enumerate(line_splitted[start_line:], start=start_line)]
        tokenized_text = tokenizer.encode(' \n'.join(numbered_lines))
        tokens_to_process = tokenized_text[:1000]
        processed_text = tokenizer.decode(tokens_to_process)
        sections = open_file(f'{seg_path}').replace('<<<<CLINICAL_NOTE>>>>', processed_text)
        if model_name == 'llama3':
            time_output_zero_shot = open_source_model_inference_llama(sections, model, tokenizer, config)
        else:
            time_output_zero_shot = open_source_model_inference(sections, model, tokenizer, config)
        try:
            time_output_zero_shot=extract_section_ranges(time_output_zero_shot)
            time_output_zero_shot=time_output_zero_shot.replace(" - ","-").replace("None","0")
            sec = extract_section_info(time_output_zero_shot)
            print("Note Segmentation Process: ", sec)
            last_section = sec['section3']
            last_2_section = sec['section2']
            sec1_text_ner = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+1])
            sections_dict_ner.append(sec1_text_ner)
            if count_while == 0:
                sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
            else: 
                if sec['section1'][0] > 3:
                    sec1_text = ' \n'.join(line_splitted[sec['section1'][0]-3:sec['section1'][1]+3])
                else:
                    sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
            sections_dict.append(sec1_text)
            sec2_text_for_ner = ' \n'.join(line_splitted[sec['section2'][0]:sec['section2'][1]+1])
            if sec['section2'][0] > 3:
                sec2_text = ' \n'.join(line_splitted[sec['section2'][0]-3:sec['section2'][1]+3])
            else:
                sec2_text = ' \n'.join(line_splitted[sec['section2'][0]:sec['section2'][1]+3])
            sections_dict.append(sec2_text)
            sections_dict_ner.append(sec2_text_for_ner)
            if last_section[0] < last_2_section[1]:
                renew_line=last_2_section[1]+1
                last_section_start_line=line_numbers[renew_line]
            else:
                last_section_start_line = line_numbers[last_section[0]]
            start_line = last_section_start_line
            count_while +=1   
            text_temp = ' '.join(line_splitted[start_line:])
            original_tokens = tokenizer.tokenize(text_temp)
        except:
            continue
    numbered_lines = [f"{i}: {line}" for i, line in enumerate(line_splitted
                                                              [start_line:], start=start_line)]
    processed_text=' \n'.join(numbered_lines)
    sections = open_file(f'{seg_path}').replace('<<<<CLINICAL_NOTE>>>>', processed_text)
    if model_name == 'llama3':
        time_output_zero_shot = open_source_model_inference_llama(sections, model, tokenizer, config)
    else:
        time_output_zero_shot = open_source_model_inference(sections, model, tokenizer, config)
    try:
        time_output_zero_shot=extract_section_ranges(time_output_zero_shot)
        time_output_zero_shot=time_output_zero_shot.replace(" - ","-")
        sec = extract_section_info(time_output_zero_shot)
        print("Note Segmentation Process: ", sec)
        sec1_text_ner = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+1])
        if count_while == 0:
            sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
        else:
            if sec['section1'][0] > 3:
                sec1_text = ' \n'.join(line_splitted[sec['section1'][0]-3:sec['section1'][1]+3])
            else:
                sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
        sections_dict.append(sec1_text)
        sections_dict_ner.append(sec1_text_ner)
        sec2_text_ner = ' \n'.join(line_splitted[sec['section2'][0]:sec['section2'][1]+1])
        if sec['section2'][0] > 3:
            sec2_text = ' \n'.join(line_splitted[sec['section2'][0]-3:sec['section2'][1]+3])
        else: 
            sec2_text = ' \n'.join(line_splitted[sec['section2'][0]-3:sec['section2'][1]+3])
        sections_dict.append(sec2_text)
        sections_dict_ner.append(sec2_text_ner)
        sec3_text = ' \n'.join(line_splitted[sec['section3'][0]:])
        sec3_text_ner = ' \n'.join(line_splitted[sec['section3'][0]-3:])
        sections_dict.append(sec3_text)
        sections_dict_ner.append(sec3_text_ner)
    except:
        print("error")
        return sections_dict,sections_dict_ner
    return sections_dict,sections_dict_ner




def extract_sections_gpt(text, hadm_id, API_MODEL, OPEN_API_KEY, API_BASE,seg_path,start_line=0):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    count_while= 0
    line_splitted = text.split('\n')
    line_numbers = list(range(len(line_splitted)))
    original_tokens = tokenizer.tokenize(text)
    sections_dict = []
    sections_dict_ner = []
    while len(original_tokens) > 1000: 
        numbered_lines = [f"{i}: {line}" for i, line in enumerate(line_splitted[start_line:], start=start_line)]
        tokenized_text = tokenizer.encode(' \n'.join(numbered_lines))
        tokens_to_process = tokenized_text[:1000]
        processed_text = tokenizer.decode(tokens_to_process)
        sections = open_file(f'{seg_path}').replace('<<<<CLINICAL_NOTE>>>>', processed_text)
        time_output_zero_shot = chatgpt_completion(sections, API_MODEL, OPEN_API_KEY, API_BASE)
        try:
            time_output_zero_shot=extract_section_ranges(time_output_zero_shot)
            time_output_zero_shot=time_output_zero_shot.replace(" - ","-").replace("None","0")
            sec = extract_section_info(time_output_zero_shot)
            print("Note Segmentation Process: ", sec)
            last_section = sec['section3']
            last_2_section = sec['section2']
            sec1_text_ner = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+1])
            sections_dict_ner.append(sec1_text_ner)
            if count_while == 0:
                sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
            else: 
                if sec['section1'][0] > 3:
                    sec1_text = ' \n'.join(line_splitted[sec['section1'][0]-3:sec['section1'][1]+3])
                else:
                    sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
            sections_dict.append(sec1_text)
            sec2_text_for_ner = ' \n'.join(line_splitted[sec['section2'][0]:sec['section2'][1]+1])
            if sec['section2'][0] > 3:
                sec2_text = ' \n'.join(line_splitted[sec['section2'][0]-3:sec['section2'][1]+3])
            else:
                sec2_text = ' \n'.join(line_splitted[sec['section2'][0]:sec['section2'][1]+3])
            sections_dict.append(sec2_text)
            sections_dict_ner.append(sec2_text_for_ner)
            if last_section[0] < last_2_section[1]:
                renew_line=last_2_section[1]+1
                last_section_start_line=line_numbers[renew_line]
            else:
                last_section_start_line = line_numbers[last_section[0]]
            start_line = last_section_start_line
            count_while +=1   
            text_temp = ' '.join(line_splitted[start_line:])
            original_tokens = tokenizer.tokenize(text_temp)
        except:
            continue
    numbered_lines = [f"{i}: {line}" for i, line in enumerate(line_splitted
                                                              [start_line:], start=start_line)]
    processed_text=' \n'.join(numbered_lines)
    sections = open_file(f'{seg_path}').replace('<<<<CLINICAL_NOTE>>>>', processed_text)
    time_output_zero_shot = chatgpt_completion(sections, API_MODEL, OPEN_API_KEY, API_BASE)
    try:
        time_output_zero_shot=extract_section_ranges(time_output_zero_shot)
        time_output_zero_shot=time_output_zero_shot.replace(" - ","-")
        sec = extract_section_info(time_output_zero_shot)
        print("Note Segmentation Process: ", sec)
        sec1_text_ner = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+1])
        if count_while == 0:
            sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
        else:
            if sec['section1'][0] > 3:
                sec1_text = ' \n'.join(line_splitted[sec['section1'][0]-3:sec['section1'][1]+3])
            else:
                sec1_text = ' \n'.join(line_splitted[sec['section1'][0]:sec['section1'][1]+3])
        sections_dict.append(sec1_text)
        sections_dict_ner.append(sec1_text_ner)
        sec2_text_ner = ' \n'.join(line_splitted[sec['section2'][0]:sec['section2'][1]+1])
        if sec['section2'][0] > 3:
            sec2_text = ' \n'.join(line_splitted[sec['section2'][0]-3:sec['section2'][1]+3])
        else: 
            sec2_text = ' \n'.join(line_splitted[sec['section2'][0]-3:sec['section2'][1]+3])
        sections_dict.append(sec2_text)
        sections_dict_ner.append(sec2_text_ner)
        sec3_text = ' \n'.join(line_splitted[sec['section3'][0]:])
        sec3_text_ner = ' \n'.join(line_splitted[sec['section3'][0]-3:])
        sections_dict.append(sec3_text)
        sections_dict_ner.append(sec3_text_ner)
    except:
        print("error")
        return sections_dict,sections_dict_ner
    return sections_dict,sections_dict_ner

def extract_occurrences(input_string, input_table):
    step4_content = re.split(r'Step 4\)', input_string, flags=re.IGNORECASE)[-1]
    mentioned_count = len(re.findall(r'Mentioned', step4_content))
    predefined_dict = {
        'chartevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'd_items': ['LABEL','LINKSTO'],
        'd_labitems': ['LABEL', 'FLUID', 'LOINC_CODE'],
        'inputevents_cv': ['CHARTTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM', 'ORIGINALROUTE'],
        'inputevents_mv': ['STARTTIME', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM'],
        'labevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'microbiologyevents': ['CHARTTIME', 'ORG_NAME', 'SPEC_TYPE_DESC'],
        'outputevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'prescriptions': ['DRUG', 'STARTDATE', 'ENDDATE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_UNIT_DISP', 'ROUTE']
    }
    occurrences = re.findall(r'Mentioned \[?\d+\]?.*?(?=Mentioned \[?\d+\]?|$)', step4_content, re.DOTALL)
    occurrences_dicts = []
    expected_fields = predefined_dict.get(input_table, [])

    for occ in occurrences:
        occ_dict = {field: 'NaN' for field in expected_fields}
        for field in expected_fields:
            pattern = fr'{field}: (.*?)(, |$|\n)'
            match = re.search(pattern, occ)
            if match and match.group(1).strip() != '':
                occ_dict[field] = match.group(1).strip() 
        occurrences_dicts.append(occ_dict)

    if not occurrences_dicts:
        occurrences_dicts = [{field: 'NaN' for field in expected_fields}]

    new_occurrences_dicts = []
    for occur in occurrences_dicts:
        nan_switch = 1
        for fie, val in occur.items():
            if 'NaN' != val and 'NaN ' not in val:
                nan_switch = 0
                break
        if nan_switch == 0:
            new_occurrences_dicts.append(occur)
            
    if len(new_occurrences_dicts) > 1 and all(occur == new_occurrences_dicts[0] for occur in new_occurrences_dicts):
        new_occurrences_dicts = [new_occurrences_dicts[0]]

    if len(new_occurrences_dicts) > 10 or len(new_occurrences_dicts) == 0:
        return [{field: 'NaN' for field in expected_fields}]
    
    return new_occurrences_dicts


def generate_formatted_output(extracted_occurrence_list, entity):
    formatted_output = ""
    question_number = 1

    for item in extracted_occurrence_list:
        for key, value in item.items():
            if key.lower() == 'label' or key.lower() == 'drug':
                clean_value = str(value).strip(string.punctuation + " ").upper()
                if clean_value != 'NAN':
                    formatted_output += f"[{question_number}] Is it directly mentioned that {entity} refers to '{value}'\n?"
                    question_number += 1
            else:
                clean_value = str(value).strip(string.punctuation + " ").upper()
                if clean_value != 'NAN':
                    formatted_output += f"[{question_number}] Is it directly mentioned that {entity}’s {key} is '{value}'?\n"
                    question_number += 1
    return formatted_output

def clean_and_eval_query(input_str):
    start = input_str.find("{'Q':")
    end = input_str.rfind("}") + 1
    if start != -1 and end != -1:
        clean_str = input_str[start:end]
        try:
            query_dict = ast.literal_eval(clean_str)
            return query_dict
        except (SyntaxError, ValueError) as e:
            return f"Error converting to dictionary: {e}"
    else:
        return "Input does not contain a valid dictionary structure."

def parse_questions_with_return(text):
    pattern = r"(?:\d+\)|\[Answer (\d+)\])(?:\:)?\s*(.+?)(?=\n(?:\d+\)|\[(Question|Answer))|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    result = {f"question{index + 1}": ' '.join(match[1].strip().split()) for index, match in enumerate(matches)}
    question1 = result.get("question1", "")
    question2 = result.get("question2", "")
    question3 = result.get("question3", "")
    
    return result, question1, question2, question3


def process_data_with_accurate_parsing(inputstring, extracted_occurence_list):
    question_answer_pairs = inputstring.split('\n[') 

    merged_dict = {}
    all_keys = set()
    for dictionary in extracted_occurence_list:
        merged_dict.update(dictionary)
        all_keys.update(dictionary.keys())
    question_answers = {}
    for pair in question_answer_pairs:  
        if 'Answers:' in pair:
            try:
                question_part, answer_part = pair.split('\nAnswer: ')
                if 'evidence quote:' in question_part.lower():
                    question_part = question_part.split('Evidence quote:')[0]
                question_part_lower = question_part.lower()
                key = next((k for k in all_keys if k.lower() in question_part_lower), None)  
                if key:
                    answer = answer_part.split('.')[0] 
                    question_answers[key] = answer
            except (IndexError, ValueError, StopIteration):
                continue  

    for key, answer in question_answers.items():
        if key in merged_dict and answer == 'No':
            merged_dict[key] = 'NaN' 

    return merged_dict


def format_input_for_tables(input_dist, specified_tables):
    predefined_dict = {
        'Chartevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'chartevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'D_items': ['LABEL','LINKSTO'],
        'd_items': ['LABEL','LINKSTO'],
        'D_labitems': ['LABEL', 'FLUID', 'LOINC_CODE'],
        'd_labitems': ['LABEL', 'FLUID', 'LOINC_CODE'],
        'Inputevents_cv': ['CHARTTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM', 'ORIGINALROUTE'],
        'inputevents_cv': ['CHARTTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM', 'ORIGINALROUTE'],
        'Inputevents_mv': ['STARTTIME', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM'],
        'inputevents_mv': ['STARTTIME', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM'],
        'Labevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'labevents': ['CHARTTIME', 'VALUENUM', 'VALUEUOM'],
        'Microbiologyevents': ['CHARTTIME', 'ORG_NAME', 'SPEC_TYPE_DESC'],
        'microbiologyevents': ['CHARTTIME', 'ORG_NAME', 'SPEC_TYPE_DESC'],
        'Outputevents': ['CHARTTIME', 'VALUE', 'VALUEUOM'],
        'outputevents': ['CHARTTIME', 'VALUE', 'VALUEUOM'],
        'Prescriptions': ['DRUG', 'STARTDATE', 'ENDDATE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_UNIT_DISP', 'ROUTE'],
        'prescriptions': ['DRUG', 'STARTDATE', 'ENDDATE', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_UNIT_DISP', 'ROUTE']
    }
    filtered_input = {k: v for k, v in input_dist.items() if v != 'NaN'}
    output = []
    for key, value in filtered_input.items():
        for table in specified_tables:
            if key in predefined_dict[table]:
                output.append(f"{table}.{key}: '{value}'")

    return output

def extract_name_item_list(string):
    lines = string.split('\n')
    result = []
    for line in lines:
        if line: 
            parts = line.split('.')
            name = parts[0]
            item = parts[1].split(':')[0]
            result.append(f"{name}.{item}")
    return result


def extract_matching_items_ignore_case(example_string, gold_list):
    gold_list_lower = [item.lower() for item in gold_list]
    example_string_corrected = example_string.replace("Prescriptions.", "\nPrescriptions.").strip()
    lines = example_string_corrected.split('\n')

    extracted_items = {}

    for line in lines:
        if '=' in line:  
            key, value = line.split('=', 1)
            key_lower = key.strip().lower()
            value_stripped = value.strip()
            if key_lower in gold_list_lower and value_stripped.lower() not in ['none', 'null']:
                extracted_items[key.strip()] = value_stripped
    result = '\n'.join(f"{key}={value}" for key, value in extracted_items.items())
    return result

def sql_selection(answer_3, res_self_corr):
    answer3_lower = answer_3.lower()

    def is_not_nan(value):
        return value is not None and str(value).lower() != 'nan'

    starttime = res_self_corr.get('STARTTIME', 'NaN').lower()
    endtime = res_self_corr.get('ENDTIME', 'NaN').lower()
    startdate = res_self_corr.get('STARTDATE', 'NaN').lower()
    enddate = res_self_corr.get('ENDDATE', 'NaN').lower()
    charttime = res_self_corr.get('CHARTTIME', 'NaN').lower()

    if is_not_nan(starttime) and is_not_nan(endtime):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if is_not_nan(startdate) and is_not_nan(enddate):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if is_not_nan(starttime) and not is_not_nan(endtime):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if is_not_nan(startdate) and not is_not_nan(enddate):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if not is_not_nan(starttime) and is_not_nan(endtime):
        return "SQL_3"
    if not is_not_nan(startdate) and is_not_nan(enddate):
        return "SQL_3"
    if not is_not_nan(starttime) and not is_not_nan(endtime) and not is_not_nan(startdate) and not is_not_nan(enddate) and not is_not_nan(charttime):
        return "SQL_0"
    if is_not_nan(charttime):
        if answer3_lower.startswith('directly written in the format yyyy-mm-dd'):
            return "SQL_1"
        elif answer3_lower.startswith('inferable time stamp from narrative'):
            return "SQL_2"

    return "SQL_0"

def sql_selection_not_dis(answer_3, res_self_corr):
    answer3_lower = answer_3.lower()

    def is_not_nan(value):
        return value is not None and str(value).lower() != 'charttime'

    starttime = res_self_corr.get('STARTTIME', 'CHARTTIME').lower()
    endtime = res_self_corr.get('ENDTIME', 'CHARTTIME').lower()
    startdate = res_self_corr.get('STARTDATE', 'CHARTTIME').lower()
    enddate = res_self_corr.get('ENDDATE', 'CHARTTIME').lower()
    charttime = res_self_corr.get('CHARTTIME', 'NaN').lower()

    if is_not_nan(starttime) and is_not_nan(endtime):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if is_not_nan(startdate) and is_not_nan(enddate):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if is_not_nan(starttime) and not is_not_nan(endtime):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if is_not_nan(startdate) and not is_not_nan(enddate):
        return "SQL_1" if answer3_lower.startswith('directly written in the format yyyy-mm-dd') else "SQL_2"
    if not is_not_nan(starttime) and is_not_nan(endtime):
        return "SQL_2"
    if not is_not_nan(startdate) and is_not_nan(enddate):
        return "SQL_2"
    if not is_not_nan(starttime) and not is_not_nan(endtime) and not is_not_nan(startdate) and not is_not_nan(enddate) and not is_not_nan(charttime):
        return "SQL_2"
    if is_not_nan(charttime):
        if answer3_lower.startswith('directly written in the format yyyy-mm-dd'):
            return "SQL_1"
        else:
            return "SQL_2"
    return "SQL_2"

def find_keys_for_tables(table_list_):
    table_list = [item.lower() for item in table_list_]
    join_dict = {
    'itemid': [['chartevents', 'd_items'], ['outputevents', 'd_items'], ['inputevents_mv', 'd_items'], ['inputevents_cv', 'd_items'], ['labevents', 'd_labitems']],
    'spec_itemid': [['microbiologyevents', 'd_items']],
    'org_itemid': [['microbiologyevents', 'd_items']]}

    found_keys = []
    for key, value in join_dict.items():
        if any(set(table_list).issubset(set(pair)) for pair in value):
            found_keys.append(key)
    return found_keys


def find_times_for_tables(table_list_):
    table_list = [item.lower() for item in table_list_]
    time_dict = {
    'charttime': [['chartevents', 'd_items'], ['outputevents', 'd_items'], ['microbiologyevents', 'd_items'], ['inputevents_cv', 'd_items'], ['labevents', 'd_labitems']],
    'starttime': [['inputevents_mv', 'd_items']],
    'endtime': [['inputevents_mv', 'd_items']],
    'startdate': [['prescriptions']],
    'enddate': [['prescriptions']]}

    found_keys = []
    for key, value in time_dict.items():
        for pair in value:
            if set(table_list).issubset(set(pair)):
                for table in table_list:
                    if table in pair:
                        found_keys.append(f"{table}.{key}")
                        break  
    return found_keys


def generate_masked_combinations(sql, condition_values):
    result_dict = {}
    
    new_condition_values = []
    
    for val in condition_values:
        splitted_val = val.split('=')
        if splitted_val[0].strip()+'='+splitted_val[1] in sql:
            new_condition_values.append(splitted_val[0].strip()+'='+splitted_val[1])
        elif splitted_val[0].strip()+"='"+splitted_val[1]+"'" in sql:
            new_condition_values.append(splitted_val[0].strip()+"='"+splitted_val[1]+"'")

    print('sql in function is ', sql)
    print('new condition values are ', new_condition_values)

    for r in range(1, len(new_condition_values) + 1):
        for combo in combinations(new_condition_values, r):
            temp_str = sql
            for item in combo:
                temp_temp_str = temp_str.replace(' AND '+item, "")
                if temp_temp_str != temp_str:
                    temp_str = temp_temp_str
                else:
                    temp_temp_str = temp_str.replace(item+' AND ', "")
                    if temp_temp_str != temp_str:
                        temp_str = temp_temp_str
                
            result_dict.setdefault(r, []).append(temp_str.strip())
    
    return 


def create_dicts(lab_id, d_items, prescriptions, d_icd_procedures, d_icd_diagnoses):
    lower_label_itemid_dict = {}
    d_item_lower_label_itemid_dict = {}
    prescriptions_itemid_dict = {}
    d_icd_procedures_dict = {}
    d_icd_diagnoses_dict = {}
    drug_list = list(set(list(prescriptions['DRUG'])))
    for i in range(len(lab_id)):
        itemid = lab_id['ITEMID'][i]
        try:
            lower_label_itemid_dict[lab_id['LABEL'][i].lower()].append(itemid)
        except:
            lower_label_itemid_dict[lab_id['LABEL'][i].lower()] = [itemid]

    for i in range(len(d_items)):
        itemid = d_items['ITEMID'][i]
        try:
            d_item_lower_label_itemid_dict[d_items['LABEL'][i].lower()].append(itemid)
        except:
            try:
                d_item_lower_label_itemid_dict[d_items['LABEL'][i].lower()] = [itemid]
            except:
                pass

    for i in range(len(drug_list)):
        itemid = i
        try:
            prescriptions_itemid_dict[drug_list[i].lower()].append(itemid)
        except:
            try:
                prescriptions_itemid_dict[drug_list[i].lower()] = [itemid]
            except:
                pass
    
    for i in range(len(d_icd_procedures)):
        icd9_code = d_icd_procedures['ICD9_CODE'][i]
        try:
            d_icd_procedures_dict[d_icd_procedures['SHORT_TITLE'][i].lower()].append(icd9_code)
        except:
            d_icd_procedures_dict[d_icd_procedures['SHORT_TITLE'][i].lower()] = [icd9_code]
        
        try:
            d_icd_procedures_dict[d_icd_procedures['LONG_TITLE'][i].lower()].append(icd9_code)
        except:
            d_icd_procedures_dict[d_icd_procedures['LONG_TITLE'][i].lower()] = [icd9_code]        


    for i in range(len(d_icd_diagnoses)):
        icd9_code = d_icd_diagnoses['ICD9_CODE'][i]
        try:
            d_icd_diagnoses_dict[d_icd_diagnoses['SHORT_TITLE'][i].lower()].append(icd9_code)
        except:
            d_icd_diagnoses_dict[d_icd_diagnoses['SHORT_TITLE'][i].lower()] = [icd9_code]
        
        try:
            d_icd_diagnoses_dict[d_icd_diagnoses['LONG_TITLE'][i].lower()].append(icd9_code)
        except:
            d_icd_diagnoses_dict[d_icd_diagnoses['LONG_TITLE'][i].lower()] = [icd9_code]   

    label_keys = list(lower_label_itemid_dict.keys())
    d_item_label_keys = list(d_item_lower_label_itemid_dict.keys())
    prescription_keys = list(prescriptions_itemid_dict.keys())
    d_icd_procedures_label_keys = list(d_icd_procedures_dict.keys())
    d_icd_diagnoses_label_keys = list(d_icd_diagnoses_dict.keys())

    return lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, d_icd_procedures_dict, d_icd_diagnoses_dict, label_keys, d_item_label_keys, prescription_keys, drug_list, d_icd_procedures_label_keys, d_icd_diagnoses_label_keys

def abbreviation_checker(A, B):
    index = 0
    B = re.sub(r'\(.*?\)', '', B)
    for char in A:
        if char == B[index]:
            index += 1
            if index == len(B):
                return True
    return False

def sql_generator(target, label_keys, lower_label_itemid_dict, lab_id, n=2, fir_thr=0.5, sec_thr=0.7):
    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list

    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in label_keys:
            try:
                p1 = cosine.get_profile(label)
                if len(label) == 1:
                    if len(sim) == 2 and label in sim:
                        final_item_id_candidates += lower_label_itemid_dict[label]
                    continue
                if len(sim) > 4 and abbreviation_checker(label, sim):
                    final_item_id_candidates += lower_label_itemid_dict[label]
                    continue
                if len(sim) > 1 and len(sim) < 5 and sim in label:
                    final_item_id_candidates += lower_label_itemid_dict[label]
                    continue
                if cosine.similarity_profiles(p0, p1) > thr:
                    final_item_id_candidates += lower_label_itemid_dict[label]
            except:
                continue
    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)
    return final_item_id_candidates, lab_id[lab_id['ITEMID'].isin(final_item_id_candidates)]

def sql_generator_d_item(target, d_item_label_keys, d_item_lower_label_itemid_dict, d_items, n=2, fir_thr=0.5, sec_thr=0.7):
    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list

    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += d_item_lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in d_item_label_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)

    return final_item_id_candidates, d_items[d_items['ITEMID'].isin(final_item_id_candidates)]


def sql_generator_d_icd(table, target, label_keys, lower_label_itemid_dict, n=2, fir_thr=0.5, sec_thr=0.7):
    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list

    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in label_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += lower_label_itemid_dict[label]
    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)
    
    if table == 'procedures':
        return final_item_id_candidates, d_icd_procedures[d_icd_procedures['ICD9_CODE'].isin(final_item_id_candidates)]
    elif table == 'diagnoses':
        return final_item_id_candidates, d_icd_diagnoses[d_icd_diagnoses['ICD9_CODE'].isin(final_item_id_candidates)]


def combine_lists(extracted_occurence_list):
    output = []
    list_length = len(extracted_occurence_list)
    if list_length == 1:
        single_list = extracted_occurence_list[0]
        for item in single_list:
            output.append([item])
    elif list_length == 2:
        first_list, second_list = extracted_occurence_list
        second_list_extended = second_list * (len(first_list) // len(second_list)) + second_list[:len(first_list) % len(second_list)]
        for first_item, second_item in zip(first_list, second_list_extended):
            combined = [first_item, second_item]
            output.append(combined)
    else:
        print("Invalid input format.")

    return output

def sql_generator_prescription(target, d_item_label_keys, d_item_lower_label_itemid_dict, drug_list, n=2, fir_thr=0.5, sec_thr=0.7):
    final_item_id_candidates = []
    abbv_similars = [target]
    one_hop_list = []

    try:
        one_hop_list += list(sets[sets.abbreviation == target]['expansion'])
    except:
        pass

    try:
        one_hop_list += list(sets[sets.expansion == target]['abbreviation'])
    except:
        pass

    abbv_similars += one_hop_list
    cosine = Cosine(n)

    for sim in abbv_similars:
        if sim == target:
            thr = fir_thr
        else:
            thr = sec_thr

        if len(sim) == 1:
            try:
                final_item_id_candidates += d_item_lower_label_itemid_dict[sim]
            except:
                pass
            continue
        p0 = cosine.get_profile(sim)
        for label in d_item_label_keys:
            p1 = cosine.get_profile(label)
            if len(label) == 1:
                if len(sim) == 2 and label in sim:
                    final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 4 and abbreviation_checker(label, sim):
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if len(sim) > 1 and len(sim) < 5 and sim in label:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]
                continue
            if cosine.similarity_profiles(p0, p1) > thr:
                final_item_id_candidates += d_item_lower_label_itemid_dict[label]

    final_item_id_candidates = list(set(final_item_id_candidates))
    final_item_id_candidates = sorted(final_item_id_candidates)

    drug_answers = [drug_list[k] for k in final_item_id_candidates]

    return drug_answers 


def total_checker(top_items,predef_map, entity_name, sql, lab_id, drug_list, d_items, lower_label_itemid_dict, d_item_lower_label_itemid_dict, prescriptions_itemid_dict, label_keys, d_item_label_keys, prescription_keys, d_icd_procedures, d_icd_diagnoses, n=2, fir_thr=0.5, sec_thr=0.7):
    def generate_sqls(entity_name, entity_set, sql):
        
        pattern = r"(\.label\s*=\s*'[^']*')"
        matches = re.findall(pattern, sql)
        new_sqls = []
        for ent in entity_set:
            if '='+entity_name in sql:
                new_sql = sql.replace('='+entity_name, '='+ent)
            elif ' = '+entity_name in sql:
                new_sql = sql.replace(' = '+entity_name, ' = '+ent)
            elif '= '+entity_name in sql:
                new_sql = sql.replace('= '+entity_name, '= '+ent)
            elif ' ='+entity_name in sql:
                new_sql = sql.replace(' ='+entity_name, ' ='+ent)
            elif "='"+entity_name+"'" in sql:
                new_sql = sql.replace("='"+entity_name+"'", "='"+ent+"'")
            elif " = '"+entity_name+"'" in sql:
                new_sql = sql.replace(" = '"+entity_name+"'", " = '"+ent+"'")
            elif "= '"+entity_name+"'" in sql:
                new_sql = sql.replace("= '"+entity_name+"'", "= '"+ent+"'")
            elif " ='"+entity_name+"'" in sql:
                new_sql = sql.replace(" ='"+entity_name+"'", " ='"+ent+"'")
            else:
                if matches:
                    new_sql = sql.replace(matches[0], ".label ='"+ent+"'")
                else:
                    if 'labevent' in sql.lower():
                        new_sql = sql + " AND d_labitems.label = '"+ent+"'"
                    if 'inputevents' in sql.lower() or 'chartevents' in sql.lower() or 'outputevents' in sql.lower():
                        new_sql = sql + " AND d_items.label = '"+ent+"'"
            new_sqls.append([ent.lower(), new_sql.lower()])
        return new_sqls
    
    search_entities_list = [entity_name.lower()]
    try:
        predef_mappings = predef_map[entity_name.lower()]
        search_entities_list.extend(predef_mappings)
    except:
        pass

    if 'Labevents' in sql:
        for search_ent in search_entities_list:
            lab_results = sql_generator(search_ent, label_keys, lower_label_itemid_dict, lab_id, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
            if len(lab_results[0]) == 0:
                continue
            try:
                from_d_lab_items = pd.concat([from_d_lab_items, lab_results[1]])
                from_d_lab_items = from_d_lab_items.drop_duplicates(subset='ITEMID')
            except:
                from_d_lab_items = lab_results[1]

        from_d_lab_items = from_d_lab_items[from_d_lab_items['ITEMID'].isin(top_items['labevents'])]
        itemid=from_d_lab_items['LABEL']
        itemid_list=list(itemid)
        return generate_sqls(entity_name.lower(), itemid_list, sql.lower())

    elif 'Diagnoses' in sql:
        for search_ent in search_entities_list:
            diagnoses_results = sql_generator_d_icd('diagnoses', d_icd_procedures, d_icd_diagnoses, search_ent, d_icd_diagnoses_label_keys, d_icd_diagnoses_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
            if len(diagnoses_results[0]) == 0:
                continue
            try:
                from_diagnoses = pd.concat([from_diagnoses, diagnoses_results[1]])
                from_diagnoses = from_diagnoses.drop_duplicates(subset='ICD9_CODE')
            except:
                from_diagnoses = diagnoses_results[1]

        short_titles=from_diagnoses['SHORT_TITLE']
        long_titles=from_diagnoses['LONG_TITLE']
        short_title_list=list(short_titles)
        long_title_list=list(long_titles)
        itemid_list = short_title_list + long_title_list
        print("generate_sqls result:    ", generate_sqls(entity_name.lower(), itemid_list, sql.lower()))
        return generate_sqls(entity_name.lower(), itemid_list, sql.lower())
    
    elif 'Procedures' in sql:
        for search_ent in search_entities_list:
            procedures_results = sql_generator_d_icd('procedures', d_icd_procedures, d_icd_diagnoses, search_ent, d_icd_procedures_label_keys, d_icd_procedures_dict, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
            if len(procedures_results[0]) == 0:
                continue
            try:
                from_procedures = pd.concat([from_procedures, procedures_results[1]])
                from_procedures = from_procedures.drop_duplicates(subset='ICD9_CODE')
            except:
                from_procedures = procedures_results[1]

        short_titles=from_procedures['SHORT_TITLE']
        long_titles=from_procedures['LONG_TITLE']
        short_title_list=list(short_titles)
        long_title_list=list(long_titles)
        itemid_list = short_title_list + long_title_list
        return generate_sqls(entity_name.lower(), itemid_list, sql.lower())

    elif 'Prescriptions' in sql:
        for search_ent in search_entities_list:
            prescription_results = list(sql_generator_prescription(search_ent, prescription_keys, prescriptions_itemid_dict, drug_list, n=n, fir_thr=fir_thr, sec_thr=sec_thr))
            if len(prescription_results) == 0:
                continue
            try:
                from_prescription.extend(prescription_results)
            except:
                from_prescription = prescription_results
        from_prescription = [ele_ for ele_ in from_prescription if ele_ in top_items['prescriptions']]
        return generate_sqls(entity_name.lower(), from_prescription, sql.lower())
 
    else:
        for search_ent in search_entities_list:
            d_item_results = sql_generator_d_item(search_ent, d_item_label_keys, d_item_lower_label_itemid_dict, d_items, n=n, fir_thr=fir_thr, sec_thr=sec_thr)
            if len(d_item_results[0]) == 0:
                continue
            try:
                from_d_items = pd.concat([from_d_items, d_item_results[1]])
                from_d_items = from_d_items.drop_duplicates(subset='ITEMID')
            except:
                from_d_items = d_item_results[1]
        
        err_catch = 1
        for table_name in ["chartevents", "inputevents_cv", "outputevents", "inputevents_mv", "microbiologyevents"]:
            if table_name in sql.lower():
                err_catch = 0
                break
        if err_catch == 1:
            assert 0, 'SQL generation ERROR'

        from_d_items = from_d_items[from_d_items['LINKSTO'].isin([table_name])]
        from_d_items = from_d_items[from_d_items['ITEMID'].isin(top_items[table_name])]
        ditemid=from_d_items['LABEL']
        ditemid_list=list(ditemid)
        return generate_sqls(entity_name.lower(), ditemid_list, sql.lower())
    
def filter_incomplete_braces(text):
    pattern = re.compile(r'\{[^{}]*\}')
    matches = pattern.findall(text)
    result = '[' + ', '.join(matches) + ']'
    return result

def parse_data(data_str):
    cleaned_str = re.sub(r'[\[\]]', '', data_str).strip()
    items = re.findall(r'\{[^}]+\}', cleaned_str)
    return items

def normalize_item(s):
    cleaned = re.sub(r'[{}]', '', s).strip().lower()
    return tuple(cleaned.split(', '))

def restore_format(t):
    return '{' + ', '.join(item.capitalize() for item in t) + '}'

def create_regex_pattern(word):
    if any(char in word for char in "[](){}.*+?^$|\\"):
        escaped_word = re.escape(word)
    else:
        escaped_word = word
    return escaped_word.replace(" ", "(?:\\s|\n)+")

def remove_overlapping_spans_correctly(spans_dict):
    spans_with_labels = [(span[key], key) for span in spans_dict for key in span]
    spans_with_labels.sort(key=lambda x: (x[0][0], -x[0][1]))
    non_overlapping = []

    for current_span, label in spans_with_labels:
        if not non_overlapping or current_span[0] > non_overlapping[-1][0][1]:
            non_overlapping.append((current_span, label))
        else:
            if current_span[1] > non_overlapping[-1][0][1]:
                non_overlapping[-1] = (current_span, label)
    return [{label: span} for span, label in non_overlapping]

def remove_invalid_labels(text, label_spans):
    updated_label_spans = []

    for label_dict in label_spans:
        for label, (start, end) in label_dict.items():
            if (start == 0 or not text[start-1].isalpha()) and (end == len(text) - 1 or not text[end+1].isalpha()):
                updated_label_spans.append(label_dict)

    return updated_label_spans

def transform_text_corrected(text, indices):
    transformed_texts = []
    def escape_special_characters(word):
        return re.escape(word)

    for index_group in indices:
        for word, (start, end) in index_group.items():
            escaped_word = escape_special_characters(word)
            replaced_word = "{{{{**{}**}}}}".format(word)
            new_text = text[:start] + replaced_word + text[end+1:]
            if "(" in word or ")" in word:
                pattern = r'(?<!\{\{\*\*)\b' + escaped_word + r'(?!\*\*\}\})'
            else:
                pattern = r'(?<!\{\{\*\*)\b' + escaped_word + r'\b(?!\*\*\}\})'
            new_text = re.sub(pattern, "", new_text)
            transformed_texts.append(new_text)
    return transformed_texts

def correct_nan_values(input_dict):
    corrected_dict = {}
    for key, value in input_dict.items():
        if value.strip('"') == 'NaN':
            corrected_dict[key] = 'NaN'
        else:
            corrected_dict[key] = value
    return corrected_dict

def process_dictionary_case_insensitive(data):
    keys_to_check = ['dose_unit_rx', 'valueuom', 'amountuom', 'rateuom']
    lower_keys_to_check = [key.lower() for key in keys_to_check]

    for key in data.keys():
        if key.lower() in lower_keys_to_check and ' ' in data[key]:
            data[key] = data[key].split(' ')[0]

    return data

def processing_before_store_data(input_list):
    new_input_list = []
    for ele_ in input_list:
        if len(ele_) > 0:
            new_input_list.append(ele_)
    
    input_list=new_input_list
    
    if input_list and input_list[0]:
        key = list(input_list[0][0].keys())[0]

        filtered_list = [[item for item in sublist if item[key]['label'] != 'No result'] for sublist in input_list]

        non_empty_lists = [l for l in filtered_list if l]
        if non_empty_lists and all(len(l) == len(non_empty_lists[0]) for l in non_empty_lists):
            or_operation_results = []
            for items in zip(*non_empty_lists):
                true_item = next((item for item in items if item[key]['label'] == 'True'), items[0])
                or_operation_results.append(true_item)
            return or_operation_results
        else:
            longest_list = max(filtered_list, key=len)
            return longest_list
    else:
        return []

def find_line_containing_word2(text,text2, indices, word):
    split_texts = text.split('\n')

    start_idx, _ = indices
    cum_length = 0
    found_line = ""

    for line in split_texts:
        line_end_idx = cum_length + len(line)
        if cum_length <= start_idx < line_end_idx:
            found_line  = line 
            break
        cum_length += len(line) + 1  
    if found_line:
        split_text2 = text2.split('\n')
        for line_num, line in enumerate(split_text2):
            if line.strip() == found_line.strip(): 
                return line_num
    return "Line containing the word not found in text2."

def remove_specific_lines(input_text):
    keywords = ['CHARTTIME', 'STARTTIME', 'ENDTIME','STARTDATE','ENDDATE']
    lines = input_text.split('\n')
    filtered_lines = [line for line in lines if all(keyword not in line for keyword in keywords)]
    return '\n'.join(filtered_lines)


def parse_input(input_text):
    time_list = ['charttime', 'starttime', 'startdate']
    value_list = ['valuenum', 'rate', 'amount','dose_val_rx']
    result = {}
    lines = input_text.split('\n')
    
    for line in lines:
        key_value = line.split('=')
        if len(key_value) == 2:
            key, value = key_value
            if any(time_key in key.lower() for time_key in time_list):
                result['time'] = value.strip()
            elif any(value_key in key.lower() for value_key in value_list):
                try:
                    result['value'] = float(value.strip())
                except ValueError:
                    pass 
    return result

def create_regex_pattern_v2(word):
    escaped_word = re.escape(word)
    pattern = escaped_word.replace("\\ ", "(?:\\s|\n)+")
    return pattern


def extract_entities_general_nur_phy(input_text):
    step7_index = input_text.find("Step 5)")
    if step7_index == -1:
        return "Step 5 not found in the text."
    step7_text = input_text[step7_index:]
    pattern = r"\{\s*'E'\s*:\s*'([^']+)',\s*'T'\s*:\s*([0-9]+)\s*\}"
    matches = re.findall(pattern, step7_text)
    entities = [{'E': match[0], 'T': int(match[1])} for match in matches]
    return entities

def remove_none_from_list(input_list):
    return [item for item in input_list if item != 'None']

def extract_entities_general_dis(input_text):
    step7_index = input_text.find("Step 7)")
    if step7_index == -1:
        return "Step 7 not found in the text."
    step7_text = input_text[step7_index:]
    pattern = r"\{\s*'E'\s*:\s*'([^']+)',\s*'T'\s*:\s*([0-9]+)\s*\}"
    matches = re.findall(pattern, step7_text)
    entities = [{'E': match[0], 'T': int(match[1])} for match in matches]
    return entities

def filter_entities(input_list):
    filtered = [entity for entity in input_list if entity['T'] in [1, 2]]
    output = [{'E': entity['E']} for entity in filtered]
    return output

def contains_charttime(dicts):
    for d in dicts:
        for value in d.keys():
            if value.lower() in ['charttime', 'starttime', 'startdate']:
                if d[value].lower() == 'nan' or d[value].lower() == 'charttime':
                    return True
    return False

def update_values_to_charttime(input_dict):
    time_concept = ['charttime', 'startdate', 'starttime']
    for key in input_dict.keys():
        if key.upper() in (tc.upper() for tc in time_concept):
            input_dict[key] = 'charttime'
    return input_dict