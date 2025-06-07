import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Student Dropout Prediction",
    page_icon="👨‍🎓"
)

@st.cache_resource
# Load model & scaler
def load_models():
    try:
        model_path = 'XGBoost.pkl'
        scaler_path = 'scaler.pkl'
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        print(f"File tidak ditemukan: {e}")
        return None, None

model, scaler = load_models()

# Mapping fitur sesuai dokumentasi dataset:
# https://archive.ics.uci.edu/dataset/697/
# predict+students+dropout+and+academic+success
marital_status_mapping = {
    "1 - Single": 1, "2 - Married": 2, "3 - Widower": 3,
    "4 - Divorced": 4, "5 – Facto Union": 5, "6 – Legally Separated": 6
}

application_mode_mapping = {
    "1 - 1st phase - general contingent": 1, 
    "2 - Ordinance No. 612/93": 2,
    "5 - 1st phase - special contingent (Azores Island)": 5, 
    "7 - Holders of other higher courses": 7,
    "10 - Ordinance No. 854-B/99": 10, 
    "15 - International student (bachelor)": 15,
    "16 - 1st phase - special contingent (Madeira Island)": 16, 
    "17 - 2nd phase - general contingent": 17,
    "18 - 3rd phase - general contingent": 18, 
    "26 - Ordinance No. 533-A/99, item b2 (Different Plan)": 26,
    "27 - Ordinance No. 533-A/99, item b3 (Other Institution)": 27, 
    "39 - Over 23 years old": 39,
    "42 - Transfer": 42, "43 - Change of course": 43,
    "44 - Technological specialization diploma holders": 44, 
    "51 - Change of institution/course": 51,
    "53 - Short cycle diploma holders": 53, 
    "57 - Change of institution/course (International)": 57
}

course_mapping = {
    "33 - Biofuel Production Technologies": 33, 
    "171 - Animation and Multimedia Design": 171,
    "8014 - Social Service (evening attendance)": 8014, 
    "9003 - Agronomy": 9003, "9070 - Communication Design": 9070, 
    "9085 - Veterinary Nursing": 9085, "9119 - Informatics Engineering": 9119, 
    "9130 - Equinculture": 9130,"9147 - Management": 9147, 
    "9238 - Social Service": 9238, "9254 - Tourism": 9254, 
    "9500 - Nursing": 9500, "9556 - Oral Hygiene": 9556, 
    "9670 - Advertising and Marketing Management": 9670,
    "9773 - Journalism and Communication": 9773, 
    "9853 - Basic Education": 9853,
    "9991 - Management (evening attendance)": 9991
}

daytime_evening_attendance_mapping = {
    "0 - Evening": 0, 
    "1 - Daytime": 1
}

previous_qualification_mapping = {
    "1 - Secondary education": 1, 
    "2 - Higher education - bachelor's degree": 2,
    "3 - Higher education - degree": 3, "4 - Higher education - master's": 4,
    "5 - Higher education - doctorate": 5, 
    "6 - Frequency of higher education": 6,
    "9 - 12th year of schooling - not completed": 9, 
    "10 - 11th year of schooling - not completed": 11,
    "12 - Other - 11th year of schooling": 12, 
    "14 - 10th year of schooling": 14,
    "15 - 10th year of schooling - not completed": 15, 
    "19 - Basic education 3rd cycle (9th/10th/11th year) or equiv": 19,
    "38 - Basic education 2nd cycle (6th/7th/8th year) or equiv": 38, 
    "39 - Technological specialization course": 39,
    "40 - Higher education - degree (1st cycle)": 40, 
    "42 - Professional higher technical course": 42,
    "43 - Higher education - master (2nd cycle)": 43
}

nationality_mapping = {
    "1 - Portuguese": 1, "2 - German": 2,
    "6 - Spanish": 6, "11 - Italian": 11,
    "13 - Dutch": 13, "14 - English": 14,
    "17 - Lithuanian": 17, "21 - Angolan": 21,
    "22 - Cape Verdean": 22, "24 - Guinean": 24,
    "25 - Mozambican": 25, "26 - Santomean": 26,
    "32 - Turkish": 32, "41 - Brazilian": 41,
    "62 - Romanian": 62, "100 - Moldova (Republic of)": 100,
    "101 - Mexican": 101, "103 - Ukrainian": 103,
    "105 - Russian": 105, "108 - Cuban": 108,
    "109 - Colombian": 109
}

mothers_qualification_mapping = {
    "1 - Secondary Education - 12th Year of Schooling or Eq": 1, 
    "2 - Higher Education - Bachelor's Degree": 2,
    "3 - Higher Education - Degree": 3, "4 - Higher Education - Master's": 4,
    "5 - Higher Education - Doctorate": 5, 
    "6 - Frequency of Higher Education": 6,
    "9 - 12th Year of Schooling - Not Completed": 9, 
    "10 - 11th Year of Schooling - Not Completed": 10,
    "11 - 7th Year (Old)": 11, "12 - Other - 11th Year of Schooling": 12,
    "14 - 10th Year of Schooling": 14, "18 - General commerce course": 18,
    "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv": 19, 
    "22 - Technical-professional course": 22,
    "26 - 7th year of schooling": 26, 
    "27 - 2nd cycle of the general high school course": 27,
    "29 - 9th Year of Schooling - Not Completed": 29, 
    "30 - 8th year of schooling": 30,
    "34 - Unknown": 34, "35 - Can't read or write": 35,
    "36 - Can read without having a 4th year of schooling": 36,
    "37 - Basic education 1st cycle (4th/5th year) or equiv": 37,
    "38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv": 38, 
    "39 - Technological specialization course": 39,
    "40 - Higher education - degree (1st cycle)": 40, 
    "41 - Specialized higher studies course": 41,
    "42 - Professional higher technical course": 42, 
    "43 - Higher Education - Master (2nd cycle)": 43,
    "44 - Higher Education - Doctorate (3rd cycle)": 44
}

fathers_qualification_mapping = {
    "1 - Secondary Education - 12th Year of Schooling or Eq": 1,
    "2 - Higher Education - Bachelor's Degree": 2,
    "3 - Higher Education - Degree": 3, "4 - Higher Education - Master's": 4,
    "5 - Higher Education - Doctorate": 5, 
    "6 - Frequency of Higher Education": 6,
    "9 - 12th Year of Schooling - Not Completed": 9, 
    "10 - 11th Year of Schooling - Not Completed": 10,
    "11 - 7th Year (Old)": 11, "12 - Other - 11th Year of Schooling": 12,
    "13 - 2nd year complementary high school course": 13, 
    "14 - 10th Year of Schooling": 14,
    "18 - General commerce course": 18, 
    "19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv": 19,
    "20 - Complementary High School Course": 20, 
    "22 - Technical-professional course": 22,
    "25 - Complementary High School Course - not concluded": 25, 
    "26 - 7th year of schooling": 26,
    "27 - 2nd cycle of the general high school course": 27, 
    "29 - 9th Year of Schooling - Not Completed": 29,
    "30 - 8th year of schooling": 30, 
    "31 - General Course of Administration and Commerce": 31,
    "33 - Supplementary Accounting and Administration": 33, "34 - Unknown": 34,
    "35 - Can't read or write": 35, 
    "36 - Can read without having a 4th year of schooling": 36,
    "37 - Basic education 1st cycle (4th/5th year) or equiv": 37, 
    "38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv": 38,
    "39 - Technological specialization course": 39, 
    "40 - Higher education - degree (1st cycle)": 40,
    "41 - Specialized higher studies course": 41, 
    "42 - Professional higher technical course": 42,
    "43 - Higher Education - Master (2nd cycle)": 43, 
    "44 - Higher Education - Doctorate (3rd cycle)": 44
}

mothers_occupation_mapping = {
    "0 - Student": 0, 
    ("1 - Representatives of the Legislative Power and Executive Bodies, "
    "Directors, Directors and Executive Managers"): 1,
    "2 - Specialists in Intellectual and Scientific Activities": 2, 
    "3 - Intermediate Level Technicians and Professions": 3,
    "4 - Administrative staff": 4,
    "5 - Personal Services, Security and Safety Workers and Sellers": 5,
    ("6 - Farmers and Skilled Workers in Agriculture, "
    "Fisheries and Forestry"): 6,
    "7 - Skilled Workers in Industry, Construction and Craftsmen": 7,
    "8 - Installation and Machine Operators and Assembly Workers": 8,
    "9 - Unskilled Workers": 9, "10 - Armed Forces Professions": 10,
    "90 - Other Situation": 90, "99 - (blank)": 99,
    "122 - Health professionals": 122, "123 - teachers": 123,
    ("125 - Specialists in information and communication technologies "
    "(ICT)"): 125,
    ("131 - Intermediate level science and engineering technicians and "
    "professions"): 131,
    ("132 - Technicians and professionals, of intermediate level of "
    "health"): 132,
    ("134 - Intermediate level technicians from legal, social, sports, "
    "cultural and similar services"): 134,
    ("141 - Office workers, secretaries in general and data "
    "processing operators"): 141,
    ("143 - Data, accounting, statistical, financial services and "
    "registry-related operators"): 143,
    "144 - Other administrative support staff": 144,
    "151 - personal service workers": 151, "152 - sellers": 152,
    "153 - Personal care workers and the like": 153,
    ("171 - Skilled construction workers and the like, except "
    "electricians"): 171,
    ("173 - Skilled workers in printing, precision instrument manufacturing, "
    "jewelers, artisans and the like"): 173,
    ("175 - Workers in food processing, woodworking, clothing and other "
    "industries and crafts"): 175,
    "191 - cleaning workers": 191,
    ("192 - Unskilled workers in agriculture, animal production, fisheries "
    "and forestry"): 192,
    ("193 - Unskilled workers in extractive industry, construction, "
    "manufacturing and transport"): 193,
    "194 - Meal preparation assistants": 194
}

fathers_occupation_mapping = {
    "0 - Student": 0,
    ("1 - Representatives of the Legislative Power and Executive Bodies, "
    "Directors, Directors and Executive Managers"): 1,
    "2 - Specialists in Intellectual and Scientific Activities": 2,
    "3 - Intermediate Level Technicians and Professions": 3,
    "4 - Administrative staff": 4,
    "5 - Personal Services, Security and Safety Workers and Sellers": 5,
    ("6 - Farmers and Skilled Workers in Agriculture, Fisheries and "
    "Forestry"): 6,
    "7 - Skilled Workers in Industry, Construction and Craftsmen": 7,
    "8 - Installation and Machine Operators and Assembly Workers": 8,
    "9 - Unskilled Workers": 9, "10 - Armed Forces Professions": 10,
    "90 - Other Situation": 90, "99 - (blank)": 99,
    "101 - Armed Forces Officers": 101, "102 - Armed Forces Sergeants": 102,
    "103 - Other Armed Forces personnel": 103,
    "112 - Directors of administrative and commercial services": 112,
    "114 - Hotel, catering, trade and other services directors": 114,
    ("121 - Specialists in the physical sciences, mathematics, engineering "
    "and related techniques"): 121,
    "122 - Health professionals": 122, "123 - teachers": 123,
    ("124 - Specialists in finance, accounting, administrative organization, "
    "public and commercial relations"): 124,
    ("131 - Intermediate level science and engineering technicians and "
    "professions"): 131,
    ("132 - Technicians and professionals, of intermediate level of "
    "health"): 132,
    ("134 - Intermediate level technicians from legal, social, sports, "
    "cultural and similar services"): 134,
    "135 - Information and communication technology technicians": 135,
    ("141 - Office workers, secretaries in general and data processing "
    "operators"): 141,
    ("143 - Data, accounting, statistical, financial services and "
    "registry-related operators"): 143,
    "144 - Other administrative support staff": 144,
    "151 - personal service workers": 151, "152 - sellers": 152,
    "153 - Personal care workers and the like": 153,
    "154 - Protection and security services personnel": 154,
    ("161 - Market-oriented farmers and skilled agricultural and animal "
    "production workers"): 161,
    ("163 - Farmers, livestock keepers, fishermen, hunters and gatherers, "
    "subsistence"): 163,
    ("171 - Skilled construction workers and the like, except "
    "electricians"): 171,
    "172 - Skilled workers in metallurgy, metalworking and similar": 172,
    "174 - Skilled workers in electricity and electronics": 174,
    ("175 - Workers in food processing, woodworking, clothing and other "
    "industries and crafts"): 175,
    "181 - Fixed plant and machine operators": 181,
    ("182 - assembly workers 183 - Vehicle drivers and mobile equipment "
    "operators"): 182,
    ("192 - Unskilled workers in agriculture, animal production, fisheries "
    "and forestry"): 192,
    ("193 - Unskilled workers in extractive industry, construction, "
    "manufacturing and transport"): 193,
    "194 - Meal preparation assistants": 194, 
    "195 - Street vendors (except food) and street service providers": 195
}

gender_mapping = {
    "Female": 0, "Male": 1
}

boolean_mapping = {
    "No": 0, "Yes": 1
}

# Application Title
st.title("👨‍🎓 Student Dropout Risk Analysis")

# Short Description
st.markdown("""
This system is designed to help predict the potential dropout risk of students  
based on personal, academic, socioeconomic, and external factors.

Please complete the form below to receive the analysis results.
""")

# Form Input
with st.form("prediction_form"):
    st.markdown("#### 🔍 Student Data Form")

    # Expander 1: Personal and Background Information
    with st.expander("📋 Personal and Background Information"):
        # Marital Status
        marital_status = st.selectbox(
            "Marital Status", options=list(marital_status_mapping.keys())
        )
        marital_status_value = marital_status_mapping[marital_status]

        # Gender
        gender = st.selectbox(
            "Gender", options=list(gender_mapping.keys())
        )
        gender_value = gender_mapping[gender]

        # Age at enrollment
        age_at_enrollment = st.number_input(
            "Age at Enrollment", min_value=15, max_value=100, value=18
        )

        # Application Mode
        application_mode = st.selectbox(
            "Application Mode", options=list(application_mode_mapping.keys())
        )
        application_mode_value = application_mode_mapping[application_mode]

        # Application Order
        application_order = st.number_input(
            "Application Order", min_value=0, max_value=9
        )

        # Course
        course = st.selectbox(
            "Course", options=list(course_mapping.keys())
        )
        course_value = course_mapping[course]

        # Daytime Evening Attendance
        daytime_evening_attendance = st.selectbox(
            "Attendance Mode", 
            options=list(daytime_evening_attendance_mapping.keys())
        )
        daytime_evening_attendance_value = (
            daytime_evening_attendance_mapping[daytime_evening_attendance]
        )

        # Previous Qualification
        previous_qualification = st.selectbox(
            "Previous Qualification", 
            options=list(previous_qualification_mapping.keys())
        )
        previous_qualification_value = (
            previous_qualification_mapping[previous_qualification]
        )

        # Previous Qualification Grade
        previous_qualification_grade = st.number_input(
            "Previous Qualification Grade", min_value=0.0, max_value=200.0
        )

        # Nationality
        nationality = st.selectbox(
            "Nationality", 
            options=list(nationality_mapping.keys())
        )
        nationality_value = nationality_mapping[nationality]

        # Mothers Qualification
        mothers_qualification = st.selectbox(
            "Mother's Qualification", 
            options=list(mothers_qualification_mapping.keys())
        )
        mothers_qualification_value = (
            mothers_qualification_mapping[mothers_qualification]
        )

        # Fathers Qualification
        fathers_qualification = st.selectbox(
            "Father's Qualification",
            options=list(fathers_qualification_mapping.keys())
        )
        fathers_qualification_value = (
            fathers_qualification_mapping[fathers_qualification]
        )

        # Mothers Occupation
        mothers_occupation = st.selectbox(
            "Mother's Occupation",
            options=list(mothers_occupation_mapping.keys())
        )
        mothers_occupation_value = (
            mothers_occupation_mapping[mothers_occupation]
        )

        # Fathers Occupation
        fathers_occupation = st.selectbox(
            "Father's Occupation",
            options=list(fathers_occupation_mapping.keys())
        )
        fathers_occupation_value = (
            fathers_occupation_mapping[fathers_occupation]
        )

    # Expander 2: Socioeconomic Status
    with st.expander("💼 Socioeconomic Status"):
        # Displaced
        displaced = st.selectbox(
            "Displaced",
            options=list(boolean_mapping.keys())
        )
        displaced_value = boolean_mapping[displaced]
        
        # Educational Special Needs
        educational_special_needs = st.selectbox(
            "Educational Special Needs",
            options=list(boolean_mapping.keys())
        )
        educational_special_needs_value = (
            boolean_mapping[educational_special_needs]
        )

        # Debtor
        debtor = st.selectbox(
            "Debtor",
            options=list(boolean_mapping.keys())
        )
        debtor_value = boolean_mapping[debtor]

        # Tuition Fees Up to Date
        tuition_fees_up_to_date = st.selectbox(
            "Tuition Fees Up to Date",
            options=list(boolean_mapping.keys())
        )
        tuition_fees_up_to_date_value = (
            boolean_mapping[tuition_fees_up_to_date]
        )

        # Scholarship Holder
        scholarship_holder = st.selectbox(
            "Scholarship Holder",
            options=list(boolean_mapping.keys())
        )
        scholarship_holder_value = boolean_mapping[scholarship_holder]

        # International
        international = st.selectbox(
            "International Student",
            options=list(boolean_mapping.keys())
        )
        international_value = boolean_mapping[international]

    # Expander 3: Academic Information
    with st.expander("🎓 Academic Information"):
        # Admission Grade
        admission_grade = st.number_input(
            "Admission Grade", min_value=0.0, max_value=200.0
        )

        # Curricular Units 1st Sem (Credited)
        cu_1st_credited = st.number_input(
            "1st Semester: Credited Units", min_value=0
        )

        # Curricular Units 1st Sem (enrolled)
        cu_1st_enrolled = st.number_input(
            "1st Semester: Enrolled Units", min_value=0
        )

        # Curricular Units 1st Sem (evaluations)
        cu_1st_evaluations = st.number_input(
            "1st Semester: Evalutions", min_value=0
        )

        # Curricular Units 1st Sem (approved)
        cu_1st_approved = st.number_input(
            "1st Semester: Approved Units", min_value=0
        )

        # Curricular Units 1st Sem (Grade)
        cu_1st_grade = st.number_input(
            "1st Semester: Average Grade", min_value=0.0, max_value=20.0
        )

        # Curricular Units 1st Sem (Without Evaluation)
        cu_1st_wo_evaluations = st.number_input(
            "1st Semester: Units without Evaluation", min_value=0
        )

        # Curricular Units 2nd Sem (Credited)
        cu_2nd_credited = st.number_input(
            "2nd Semester: Credited Units", min_value=0
        )
        
        # Curricular Units 2nd Sem (enrolled)
        cu_2nd_enrolled = st.number_input(
            "2nd Semester: Enrolled Units", min_value=0
        )

        # Curricular Units 2nd Sem (evaluations)
        cu_2nd_evaluations = st.number_input(
            "2nd Semester: Evalutions", min_value=0
        )

        # Curricular Units 2nd Sem (approved)
        cu_2nd_approved = st.number_input(
            "2nd Semester: Approved Units", min_value=0
        )

        # Curricular Units 2nd Sem (Grade)
        cu_2nd_grade = st.number_input(
            "2nd Semester: Average Grade", min_value=0.0, max_value=20.0
        )

        # Curricular Units 2nd Sem (Without Evaluation)
        cu_2nd_wo_evaluations = st.number_input(
            "2nd Semester: Units without Evaluation", min_value=0
        )

    # Expander 4: External Economic Factors
    with st.expander("🌍 External Economic Factors"):
        # Unemployment Rate
        unemployment_rate = st.number_input(
            "Unemployment Rate (%)", min_value=0.0, max_value=100.00
        )

        # Inflation Rate
        inflation_rate = st.number_input(
            "Inflation Rate (%)", min_value=0.0, max_value=100.00
        )

        # GDP
        gdp = st.number_input("Gross Domestic Product (GDP)", min_value=0.0)

    # Submit Button
    submit = st.form_submit_button("Predict Dropout")

if submit and model is not None and scaler is not None:
    try:
        # Input Data
        data_input = pd.DataFrame({
            "Marital_status": [marital_status_value],
            "Application_mode": [application_mode_value],
            "Application_order": [application_order],
            "Course": [course_value],
            "Daytime_evening_attendance": [daytime_evening_attendance_value],
            "Previous_qualification": [previous_qualification_value],
            "Previous_qualification_grade": [previous_qualification_grade],
            "Nacionality": [nationality_value],
            "Mothers_qualification": [mothers_qualification_value],
            "Fathers_qualification": [fathers_qualification_value],
            "Mothers_occupation": [mothers_occupation_value],
            "Fathers_occupation": [fathers_occupation_value],
            "Admission_grade": [admission_grade],
            "Displaced": [displaced_value],
            "Educational_special_needs": [educational_special_needs_value],
            "Debtor": [debtor_value],
            "Tuition_fees_up_to_date": [tuition_fees_up_to_date_value],
            "Gender": [gender_value],
            "Scholarship_holder": [scholarship_holder_value],
            "Age_at_enrollment": [age_at_enrollment],
            "International": [international_value],
            "Curricular_units_1st_sem_credited": [cu_1st_credited],
            "Curricular_units_1st_sem_enrolled": [cu_1st_enrolled],
            "Curricular_units_1st_sem_evaluations": [cu_1st_evaluations],
            "Curricular_units_1st_sem_approved": [cu_1st_approved],
            "Curricular_units_1st_sem_grade": [cu_1st_grade],
            "Curricular_units_1st_sem_without_evaluations": [
                cu_1st_wo_evaluations
            ],
            "Curricular_units_2nd_sem_credited": [cu_2nd_credited],
            "Curricular_units_2nd_sem_enrolled": [cu_2nd_enrolled],
            "Curricular_units_2nd_sem_evaluations": [cu_2nd_evaluations],
            "Curricular_units_2nd_sem_approved": [cu_2nd_approved],
            "Curricular_units_2nd_sem_grade": [cu_2nd_grade],
            "Curricular_units_2nd_sem_without_evaluations": [
                cu_2nd_wo_evaluations
            ],
            "Unemployment_rate": [unemployment_rate],
            "Inflation_rate": [inflation_rate],
            "GDP": [gdp]
        })

        data_input['approval_ratio'] = (
            data_input['Curricular_units_1st_sem_approved'] + 
            data_input['Curricular_units_2nd_sem_approved']
        ) / (
            data_input['Curricular_units_1st_sem_enrolled'] + 
            data_input['Curricular_units_2nd_sem_enrolled'] + 1e-5
        )
        data_input['avg_grade_all_sem'] = (
            data_input['Curricular_units_1st_sem_grade'] + 
            data_input['Curricular_units_2nd_sem_grade']
        ) / 2

        data_input = data_input[scaler.feature_names_in_]
        # Scale features
        data_input_scaled = scaler.transform(data_input)

        # Make prediction
        prediction = model.predict(data_input_scaled)
        probability = model.predict_proba(data_input_scaled)[0][1]

        st.markdown("""
            <style>
            .custom-box {
                border-radius: 15px;
                padding: 25px;
                background: linear-gradient(to right, #f9f9f9, #e0e0e0);
                box-shadow: 4px 4px 20px rgba(0,0,0,0.1);
                text-align: center;
                margin-bottom: 25px;
                color: #222222;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .high {
                border-left: 10px solid #d62828;
                background: linear-gradient(to right, #fce8e6, #f9c6c3);
                color: #7f1d1d;
            }
            .low {
                border-left: 10px solid #2a9d8f;
                background: linear-gradient(to right, #dff7f3, #b3ede6);
                color: #14594b;
            }
            .status-title {
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 12px;
            }
            .probability-text {
                font-size: 20px;
                color: inherit;
                font-style: italic;
            }
            .badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 12px;
                background-color: rgba(255,255,255,0.6);
                font-size: 14px;
                margin-top: 10px;
                color: #555;
                font-weight: 600;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display prediction status
        status_kelas = "high" if prediction[0] == 1 else "low"
        status_text = "⚠️ Dropout Risk Detected" if prediction[0] == 1 else "✅ Low Dropout Risk"
        badge_text = "Intervention Needed" if prediction[0] == 1 else "Stable"
        prob_val = f"{probability:.2%}" if prediction[0] == 1 else f"{1 - probability:.2%}"
        desc_text = "Estimated dropout probability: " if prediction[0] == 1 else "Likely to stay enrolled: "

        st.markdown(f"""
            <div class='custom-box {status_kelas}'>
                <div class='status-title'>{status_text}</div>
                <div class='probability-text'>{desc_text}<strong>{prob_val}</strong></div>
                <div class='badge'>{badge_text}</div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as error:
        st.error("Terjadi kesalahan saat memproses data.")
        st.error(f"Rincian kesalahan: {error}")