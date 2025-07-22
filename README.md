# END-TO-END-DATA-SCIENCE-PROJECT

COMPANY : CODTECH IT SOLUTIONS 
NAME : AMOL SHIVAJI KADAM 
INTERN ID : CT08DN391 
DOMAIN : DATA SCIENCE 
DURARION : 8 WEEKS 
MENTOR : NEELA SANTOSH

# Description
This project involves the development of a web-based machine learning application to predict the risk of diabetes using health-related input data. The application is designed with an intuitive front-end interface and a robust back-end built with FastAPI. A trained Random Forest machine learning model forms the core of the system, providing real-time predictions based on user input. The application not only performs predictions but also educates users through health tips, feedback, and personalized suggestions, offering a comprehensive digital health assessment experience.

# Project Components and Workflow:

# Machine Learning Model (model.py):

The machine learning component uses the Random Forest Classifier from scikit-learn. The model is trained on the Pima Indians Diabetes Dataset, a widely used dataset in the healthcare domain. This dataset includes eight key features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age. These features are used to predict the binary outcome of whether a person is diabetic or not.

Once trained, the model is serialized using Python’s pickle library and saved to a file named diabetes_model.pkl. A function load_model() is implemented to load the model for use during inference in the FastAPI server.

# FastAPI Backend (main.py):

The backend is built using FastAPI, a high-performance web framework. It includes several important endpoints:

Home ("/"): Displays the HTML form to the user where health parameters are entered.

Predict ("/predict-form"): Accepts form submissions, processes input data, performs prediction using the trained model, stores the data with timestamp in a CSV file, and redirects the user to the result page.

Result ("/result"): Shows the prediction outcome along with input details and recommendations.

The backend also uses pandas for CSV handling and logging user submissions, including input features, prediction result, and timestamp (formatted according to Indian local time).

# Data Schema (schemas.py):

This file contains the DiabetesInput class defined using Pydantic's BaseModel. It validates the structure and types of the input fields, ensuring data integrity before prediction.

Front-End Interface (form.html and result.html):

The frontend is crafted using HTML and styled with Tailwind CSS to provide a modern, responsive user experience.

form.html: This file presents a user-friendly form with animated visuals, floating health icons, glowing buttons, and progress indicators. It includes client-side scripts that help users understand their glucose and BMI levels in real-time as they enter values. It also offers daily health tips, motivational quotes, and risk factor awareness to educate users.

result.html: This file displays the results after prediction. Depending on whether the risk is high or low, the visual elements adjust accordingly with different colors, messages, and animations. Users receive actionable advice, such as consulting a doctor or maintaining a healthy lifestyle. There is also an option to print, share, or perform a new analysis. Confetti animations are triggered for non-diabetic results, creating a positive user experience.

# Prediction Storage:

Every user prediction is appended to a local CSV file (user_data.csv) with a timestamp. This allows for historical tracking of predictions and can later be used for analytics or audits. The system ensures the file is created if it doesn’t exist and appends new entries in a structured format using pandas.

# Key Features and Benefits:

Accurate AI Predictions: Utilizes a robust Random Forest model trained on medical data for reliable risk classification.

Interactive UI: Engaging animations, typewriter effects, live feedback indicators for glucose and BMI, and health stats counters enhance user engagement.

Health Awareness: The tool provides real-time tips and educates users on risk factors and lifestyle management.

User-Centric Results: Based on the prediction, users are offered personalized suggestions, with clear visual indicators of risk.

Data Logging: Ensures each user interaction is recorded with relevant data and time, supporting further use in analytics or performance tracking.

# Technologies Used:

Python for model development and backend logic

FastAPI as the web server framework

Pandas and Pickle for data handling and model serialization

HTML, Tailwind CSS, and JavaScript for frontend design

Pydantic for input data validation

# Impact and Applicability:

This AI health prediction tool can be effectively used in health camps, clinics, colleges, and community awareness programs. It acts as a low-cost, scalable, and accessible solution for preliminary diabetes screening. The engaging design and reliable backend make it suitable for public use while maintaining the accuracy and seriousness required in healthcare-related tools.

# Conclusion:

This project successfully integrates machine learning with web development to deliver a health-focused, user-friendly application. The combination of accurate predictions, dynamic interface, and user education makes it a valuable tool for health awareness and screening. Its modular structure allows for easy expansion into mobile apps, multilingual platforms, or integration with wearable health devices in the future.

# Output
https://github.com/user-attachments/assets/7444614d-c7f9-4094-8ce2-2634dfd6bede   
https://github.com/user-attachments/assets/a352bd5b-6e73-401b-b75f-6dd34bb01105
