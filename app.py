from joblib import load
from flask import Flask, request, jsonify, render_template
from model_helper import load_model_and_preprocessors,get_top_recommendations,predict_with_model
import numpy as np
# Load the saved model
model = load("xgr_calories.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/calories')
def calories_page():
    return render_template('calories.html')

@app.route('/diet')
def diet_page():
    return render_template('diet.html')

@app.route('/exercise')
def exercise_page():
    return render_template('exercise.html')

@app.route('/predict_calories', methods=['POST'])
def predict_calories():
    try:
        data = request.json
        gender = data['gender']  # 0 or 1
        heartrate = data['heartrate']
        body_temp = data['body_temp']
        height = data['height']
        age = data['age']
        
        inputs = [[float(gender), float(age), float(height), float(heartrate), float(body_temp)]]
        calories = model.predict(inputs)[0]
        
        return jsonify({'calories': float(calories)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_diet', methods=['POST'])
def predict_diet():
    try:
        data = request.json
        inputs =np.asarray( [[data['Age'], data['Height'], data['Weight'], data['BMI'],
                  data['Sex'], data['Level'], data['Fitness Goal'], data['Fitness Type'],
                  data['Hypertension'], data['Diabetes']]])
        path="models/gym_diet"
        model_name="diet"
        model, mlb, scaler,label_dict=load_model_and_preprocessors(path,model_name)
        diet_recommendations = predict_with_model(model, scaler, mlb, inputs, top_n=3,label_encoder=label_dict)
        final=diet_recommendations[0]
        out=""
        for i in final:
            out=out+"    "+i
        
        
        return jsonify({'diet_recommendation': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_exercise', methods=['POST'])
def predict_exercise():
    try:
        data = request.json
        inputs = np.asarray([[data['Age'], data['Height'], data['Weight'], data['BMI'],
                  data['Sex'], data['Level'], data['Fitness Goal'], data['Fitness Type'],
                  data['Hypertension'], data['Diabetes']]])
        print(inputs.shape)
        path="models/gym_diet"
        model_name="exercise"
        model, mlb, scaler,label_dict=load_model_and_preprocessors(path,model_name)
        exer_recommendations = predict_with_model(model, scaler, mlb, inputs, top_n=3,label_encoder=label_dict)
        final=exer_recommendations[0]
        out=""
        for i in final:
            out=out+"    "+i
        print("Diet recommendations for sample inputs:", out)
        
        
        return jsonify({'exercise_recommendation': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
