# Data

## Reference dataset (not included)

Heart Disease Dataset (Comprehensive), IEEE Dataport, https://doi.org/10.21227/dz4t-cm36. Save the file as `data/real_data.csv`. It contains 1,190 records, of which 918 are unique; 272 Statlog records duplicate Cleveland records. Missing cholesterol values are coded as 0 in 172 records.

## Synthetic cohorts

`synthetic/<model>_run<k>.csv`, twelve files generated as described in the main README. The cohorts contain no information beyond what the language models returned. In `gemini_run1.csv`, 84 of the 192 records are identical to records of the public reference dataset in all 12 fields; these records are kept unchanged because their presence is one of the findings of the study. The reference dataset is itself public and de-identified.

The synthetic cohorts are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Please cite the article and the reference dataset when using them.

## Variables

| Column | Description | Coding |
| --- | --- | --- |
| `age` | age | years |
| `sex` | sex | 1 male, 0 female |
| `chest pain type` | chest pain type | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `resting bp s` | resting blood pressure | mm Hg |
| `cholesterol` | serum cholesterol | mg/dl |
| `fasting blood sugar` | fasting blood sugar > 120 mg/dl | 1 true, 0 false |
| `resting ecg` | resting electrocardiogram | 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy |
| `max heart rate` | maximum heart rate achieved | beats per minute |
| `exercise angina` | exercise-induced angina | 1 yes, 0 no |
| `oldpeak` | ST depression induced by exercise | mm |
| `ST slope` | slope of the peak exercise ST segment | 1 upsloping, 2 flat, 3 downsloping |
| `target` | heart disease | 1 present, 0 absent |
