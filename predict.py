import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
csv_file = "C:\\Users\\sprab\\Ml_lab\\APEAPCET2023LASTRANKDETAILS.csv"
df = pd.read_csv(csv_file)

# Replace non-numeric values in numeric columns with NaN
numeric_columns = ['COLLEGE FEE'] + [
    'OC_BOYS', 'OC_GIRLS', 'SC_BOYS', 'SC_GIRLS', 'BCA_BOYS', 'BCA_GIRLS',
    'BCB_BOYS', 'BCB_GIRLS', 'BCC_BOYS', 'BCC_GIRLS', 'BCD_BOYS', 'BCD_GIRLS',
    'BCE_BOYS', 'BCE_GIRLS', 'OC_EWS_BOYS', 'OC_EWS_GIRLS', 'ST_BOYS', 'ST_GIRLS'
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') 

df.fillna(0, inplace=True)

label_encoders = {}
categorical_columns = ['INSTCODE', 'NAME OF THE INSTITUTION', 'INST_REG', 'BRANCH_CODE', 'TYPE']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


category_columns = [
    'OC_BOYS', 'OC_GIRLS', 'SC_BOYS', 'SC_GIRLS', 'BCA_BOYS', 'BCA_GIRLS',
    'BCB_BOYS', 'BCB_GIRLS', 'BCC_BOYS', 'BCC_GIRLS', 'BCD_BOYS', 'BCD_GIRLS',
    'BCE_BOYS', 'BCE_GIRLS', 'OC_EWS_BOYS', 'OC_EWS_GIRLS', 'ST_BOYS', 'ST_GIRLS'
]
target_columns = ['INSTCODE', 'NAME OF THE INSTITUTION', 'COLLEGE FEE', 'TYPE']
features = df[['BRANCH_CODE', 'INST_REG'] + category_columns]
targets = df[target_columns]
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(features, targets)

def predict_institutions(person_rank, category_gender, branch_code, inst_reg):
    category_gender = category_gender.upper().strip()
    branch_code = branch_code.upper().strip()
    inst_reg = inst_reg.upper().strip()

    if category_gender not in [col.upper() for col in category_columns]:
        raise ValueError(f"Invalid category_gender: {category_gender}. Must be one of {category_columns}")

    if branch_code not in label_encoders['BRANCH_CODE'].classes_:
        raise ValueError(
            f"Invalid branch_code: {branch_code}. Choose from {list(label_encoders['BRANCH_CODE'].classes_)}")

    if inst_reg not in label_encoders['INST_REG'].classes_:
        raise ValueError(f"Invalid INST_REG: {inst_reg}. Choose from {list(label_encoders['INST_REG'].classes_)}")

    branch_code_encoded = label_encoders['BRANCH_CODE'].transform([branch_code])[0]
    inst_reg_encoded = label_encoders['INST_REG'].transform([inst_reg])[0]
    input_data = [branch_code_encoded, inst_reg_encoded] + [0] * len(category_columns)
    category_index = [col.upper() for col in category_columns].index(category_gender)
    input_data[category_index + 2] = person_rank  
    predictions = model.predict([input_data])

    df['PREDICTED_DISTANCE'] = np.linalg.norm(targets - predictions, axis=1)
    top_2_colleges = df.nsmallest(2, 'PREDICTED_DISTANCE')

    results = []
    for _, row in top_2_colleges.iterrows():
        decoded_results = {
            'INSTCODE': label_encoders['INSTCODE'].inverse_transform([int(round(row['INSTCODE']))])[0],
            'NAME OF THE INSTITUTION': label_encoders['NAME OF THE INSTITUTION'].inverse_transform(
                [int(round(row['NAME OF THE INSTITUTION']))])[0],
            'COLLEGE FEE': int(round(row['COLLEGE FEE'])),
            'TYPE': label_encoders['TYPE'].inverse_transform([int(round(row['TYPE']))])[0]
        }
        results.append(decoded_results)

    return results


person_rank = int(input("Enter person rank: "))
category_gender = input(f"Enter category_gender (options: {', '.join(category_columns)}): ")
branch_code = input(f"Enter branch code (options: {', '.join(label_encoders['BRANCH_CODE'].classes_)}): ")
inst_reg = input(f"Enter INST_REG (options: {', '.join(label_encoders['INST_REG'].classes_)}): ")

try:
    results = predict_institutions(person_rank, category_gender, branch_code, inst_reg)
    print("\nPredicted Colleges:")
    for i, result in enumerate(results, 1):
        print(f"\nCollege {i}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
except Exception as e:
    print(f"Error occurred: {e}")
