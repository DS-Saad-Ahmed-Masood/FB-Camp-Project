import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import streamlit as st
import random

# Set page title and icon
st.set_page_config(
page_title="FB Campaign",
page_icon="📢",layout="wide"
)

df = pd.read_excel('Social_FB.xlsx')

#X = df.drop(columns=['approved_conversion'])
X = df.drop(columns=['approved_conversion','ad_id','xyz_campaign_id','fb_campaign_id'])
y = df['approved_conversion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['gender', 'ad_region']
categorical_features = [feature for feature in categorical_features if feature in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier(random_state=42))])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100

def preprocess_input(user_input):
    '''Convert user input to DataFrame'''
    user_df = pd.DataFrame(user_input, index=[0])
    return user_df

# Function to make predictions
def get_suggestions(user_input):
    '''Preprocess user input'''
    user_df = preprocess_input(user_input)
    prediction = pipeline.predict(user_df)
    return prediction

def generate_suggestion(prediction, user_input):
    if prediction == 1:
        approved_rules = {
            'gender == "F"': "Targeting female audiences may lead to higher approval rates.",
            'spent > 100': "Increasing the ad spend to over $100 can improve the chances of approval.",
            'interest <= 20': "Avoid targeting audiences with low interests as they may not engage with the ad.",
            'age <= 30': "Younger demographics may respond better to the ad, increasing the chances of approval.",
            'ad_region in [1, 2, 6]': "Targeting regions like Karachi, Hyderabad, and Peshawar may yield higher approval rates.",            
            'campaign_date.weekday() in [0, 1, 2, 3, 4]': "Weekdays (Monday to Friday) are generally better for running ad campaigns and may result in higher approval rates.",
            'age > 18 and age <= 25': "Targeting younger age groups between 18 and 25 may increase the likelihood of approval.",
            'spent > 50 and interest > 40': "Higher spending combined with targeting audiences with high interests can lead to better approval rates."
        }

        applicable_suggestions = [suggestion for rule, suggestion in approved_rules.items() if eval(rule, globals(), user_input)]
        if applicable_suggestions:
            return random.choice(applicable_suggestions)
        else:
            return "Your ad campaign is likely to be approved."
    else:
        not_approved_rules = {
            'age > 60': "Avoid targeting older age groups above 60 as they may not respond well to the ad.",
            'ad_region == 4': "Advertising in Islamabad may lead to lower approval rates.",
            'spent <= 20': "Higher spending is often necessary for better campaign performance. Consider increasing your budget.",
            'interest > 60': "Targeting audiences with extremely high interests may lead to lower approval rates as they may be less receptive to ads.",
            'campaign_date.weekday() in [5, 6]': "Weekends (Saturday and Sunday) may not be ideal for running ad campaigns and may result in lower approval rates.",
            'age > 25 and age <= 40': "Avoid targeting age groups between 25 and 40 as they may not be the ideal audience for the ad.",
            'spent > 20 and interest <= 10': "Lower spending combined with targeting audiences with low interests can lead to lower approval rates.",
            'ad_region in [3, 5]': "Regions like Lahore and Quetta may have lower approval rates compared to other regions."
        }

        applicable_suggestions = [suggestion for rule, suggestion in not_approved_rules.items() if eval(rule, globals(), user_input)]
        if applicable_suggestions:
            return random.choice(applicable_suggestions)
        else:
            return "Your ad campaign might not be approved."

def main():
    # # Set page title and icon
    # st.set_page_config(
    # page_title="FB Campaign",
    # page_icon="📢",layout="wide"
    # )
    st.title('Live Prediction Of Expected Conversion')

    with st.container():
        col1,col2= st.columns(spec=[0.3,0.7], gap="small")

    with col2:
        st.image("https://www.socialchamp.io/wp-content/uploads/2021/08/Feature-Banner_JulyOnwards-Q3-2021_1125x600_04.png.webp",use_column_width=True)
        st.header("Welcome to My Facebook Campaign Conversion Prediction App!")
        st.write("This application is designed to help you predict the success of your Facebook ad campaigns in terms of conversions. By providing details about your ad campaign, such as ad ID, campaign ID, age, gender, interests, region, and the amount spent, our app utilizes a (Machine Learning) model to forecast whether your campaign is likely to result in conversions or not.")
        st.write("Here's how it works:")

        st.subheader("1.Input Your Campaign Details:")
        st.write("Enter the required information about your Facebook ad campaign, including ad ID, campaign ID, age, gender, interests, region, and the amount spent.")
        
        st.subheader("2.Get Predictions:")
        st.write("Once you've entered the campaign details, our machine learning model will analyze the data and provide you with a prediction on whether your campaign is likely to be successful or not.")
        
        st.subheader("3.Make Informed Decisions:")
        st.write("Based on the prediction, you can make informed decisions about your Facebook ad campaigns, optimizing your marketing strategies for better results.")
        st.write("Whether you're a marketer, advertiser, or business owner, this Facebook Campaign Conversion Prediction App is here to help you enhance the effectiveness of your advertising efforts and achieve your marketing goals. Start predicting the success of your Facebook ad campaigns today!")
        st.success("Created by:**Mr. Saad Ahmed Masood**")

    with col1:
    # Get user input
        user_input = {
            'ad_id': st.text_input('Enter Ad ID'),
            'xyz_campaign_id': st.selectbox('Select XYZ Campaign ID', [916, 936, 1178]),
            'fb_campaign_id': st.text_input('Enter FB Campaign ID'),
            'campaign_date': st.date_input('Select Campaign Date'),
            'age': st.slider('Select Age', 18, 65, 30),
            'gender': st.selectbox('Select Gender', ['M', 'F']),
            'interest': st.slider('Select Interest', 1, 70, 30),
            'ad_region': st.selectbox("Select ad region", ["Karachi", "Hyderabad", "Lahore", "Islamabad", "Quetta", "Peshawar"]),        
            'spent': st.number_input('Enter Amount Spent $')
        }

        # Map ad region text back to numerical values cause original and training data-set is in numeric form, using for ad_region dropdown box
        region_mapping = {"Karachi": 1, "Hyderabad": 2, "Lahore": 3, "Islamabad": 4, "Quetta": 5, "Peshawar": 6}
        user_input['ad_region'] = region_mapping.get(user_input.get('ad_region'))

        if st.button('Predict'):
            # Make prediction
            prediction = get_suggestions(user_input)
            st.write('Predicted Approval:', prediction)
            
            # Provide practical segmentation suggestions based on the prediction
            suggestion = generate_suggestion(prediction[0], user_input)
            st.write('Suggestion:', suggestion)

if __name__ == "__main__":
    main()
