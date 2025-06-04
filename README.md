
****EAPCET College Predictor using Random Forest****

This project is a **machine learning-based EAPCET College Predictor** that helps students estimate the possible colleges they can get based on their **rank**, **category**, **gender**, **branch preference**, and **institution region**.
**** Dataset****

The model is trained on the **APEAPCET 2023 Last Rank Details** dataset. It contains:

- Institution details (`INSTCODE`, `NAME OF THE INSTITUTION`, `INST_REG`)
- Branch code and type
- Category and gender-wise last ranks
- College fee structure

****Source file****: `APEAPCET2023LASTRANKDETAILS.csv`

****Technologies Used****

- **Python 3**
- **Pandas** for data manipulation
- **NumPy** for numerical computation
- **Scikit-learn** for model training (Random Forest Regressor)
**** How It Works****

1. **Data Preprocessing**:
   - Converts rank columns to numeric and fills missing values.
   - Encodes categorical columns using `LabelEncoder`.

2. **Model Training**:
   - Trains a `RandomForestRegressor` using category and institution info to learn rank-college mappings.

3. **Prediction Function**:
   - User inputs:
     - Rank
     - Category & Gender (e.g., `OC_BOYS`, `SC_GIRLS`, etc.)
     - Branch Code
     - Institution Region
   - The model estimates suitable colleges based on input, using Euclidean distance to find top 2 closest matches.
**** Example Inputs****

```bash
Enter person rank: 15432
Enter category_gender (options: OC_BOYS, SC_GIRLS, ...): BCB_BOYS
Enter branch code (options: CSE, ECE, ...): CSE
Enter INST_REG (options: AU, SVU, ...): AU
