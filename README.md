# EHRCon: Dataset for Checking Consistency between Unstructured Notes and Structured Tables in Electronic Health Records
*Introduce a new task and dataset called EHRCon, designed to verify the consistency between clinical notes and large-scale relational databases in electronic health records (EHRs).*

## Updates
[2024/06] We released our paper on [arXiv](https://arxiv.org/pdf/2406.16341).

## Abstract
Electronic Health Records (EHRs) are integral for storing comprehensive patient
medical records, combining structured data (e.g., medications) with detailed clinical
notes (e.g., physician notes). These elements are essential for straightforward data
retrieval and provide deep, contextual insights into patient care. However, they often
suffer from discrepancies due to unintuitive EHR system designs and human errors,
posing serious risks to patient safety. To address this, we developed EHRCon, a new
dataset and task specifically designed to ensure data consistency between structured
tables and unstructured notes in EHRs. EHRCon was crafted in collaboration with
healthcare professionals using the MIMIC-III EHR dataset, and includes manual
annotations of 3,943 entities across 105 clinical notes checked against database
entries for consistency. EHRCon has two versions, one using the original MIMICIII schema, and another using the OMOP CDM schema, in order to increase its
applicability and generalizability. Furthermore, leveraging the capabilities of large
language models, we introduce CheckEHR, a novel framework for verifying the
consistency between clinical notes and database tables. CheckEHR utilizes an
eight-stage process and shows promising results in both few-shot and zero-shot
settings. 

## Setup
```bash
conda create -n ehrcon python=3.8
conda activate ehrcon
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Dataset

### Access Requirements and Download the dataset
The EHRCon dataset is derived from MIMIC-III, which requires a credentialed Physionet license for access. Due to these requirements and adherence to the Data Use Agreement (DUA), only credentialed users can access the MIMIC-III dataset files. To gain access to the MIMIC-III and EHRCon dataset, you must meet all of the following criteria:

1. Be a [credentialed user](https://physionet.org/settings/credentialing/)
    - If you don't have a PhysioNet account, sign up for one [here](https://physionet.org/register/).
    - Follow these [instructions](https://physionet.org/login/?next=/credential-application/) to get credentialed on PhysioNet.
    - Complete the "CITI Data or Specimens Only Research" [training](https://physionet.org/about/citi-course/) course.

2. Sign the data use agreement (DUA) for each project and download the dataset.
    - MIMIC-III: https://physionet.org/content/mimiciii/1.4/
    - EHRCon: https://physionet.org/content/ehrcon-consistency-of-notes/1.0.0/

### Build dataset and database directory
```bash
    cp ./EHRCon EHRCon/dataset/
    mkdir EHRCon/dataset/database
    cd EHRCon/dataset/database
    python EHRCon/make_database.py 
```

### Dataset Structure
```bash
EHRCon
├── dataset
    ├── original
        ├── test
            ├── discharge
                ├── discharge_label
                    └── _.pkl
                └── discharge_test.csv
            ├── nursing
            └── physician
        └── valid
    └── processed
└── database
    └── _.db
```
- ```EHRCon``` is the root directory. Within this directory, there are two main subdirectories: ```dataset``` and ```database```. 
- The ```dataset``` directory contains files related to the consistency check between clinical notes and EHR data. This directory is further divided into ```original``` and ```processed``` subdirectories. 
- The ```original``` directory includes data for consistency checks with MIMIC original notes. The ```processed``` directory contains data that has been filtered to remove information not present in the EHR tables.
- All directories (```original``` and ```processed```) within dataset follow the same structure, containing test and valid subdirectories, which further include ```discharge```, ```nursing```, and ```physician``` directories with relevant label (.pkl) and CSV files.
- ```database``` directory contains individual patient database files.

### EHRCon
EHRCon contains the following fields for each database:
- ```Row_id```: The ROW_ID of the clinical note
- ```entity```: The entity that is the subject of the consistency check
- ```data```:The values related to the entity extracted from the clinical note and mapped to the columns of the table
- ```label```: Indicates whether the content related to the entity in the note is consistent with the table, with details on which columns have inconsistencies, if any are found.
- ```errors```: The number of columns with inconsistencies
- ```position```: The line number of the entity in the note
- ```source```: mimic
- ```entity type```:
    - 1 - entities with numerical values
    - 2 - entities without values but who use existence can be verified in the database
    - 3 - entities with string values
- ```all_queries```: A list of all value pairs checked to determine the label for the entity

```json
{"Row_id": [{"entity": {"data": [{"table_name1": {"column_name1": "value"},
      "label": "'charttime' and 'valuenum' are inconsistency",
      "errors": 2}],
    "position": "4",
    "source": "mimic",
    "entity_type": "1",
    "all_queries":[]}}]}
```
## CheckEHR

### few shot
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model model_name --config_yaml /path/to/config
```
### zero shot
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model model_name --config_yaml /path/to/config --mode zero_shot
```

## Question
If you have any questions, contact us at ([yeonsu.k@kaist.ac.kr](mailto:yeonsu.k@kaist.ac.kr) or [jiho.kim@kaist.ac.kr](mailto:jiho.kim@kaist.ac.kr))
