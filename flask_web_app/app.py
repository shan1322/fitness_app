from joblib import load
from flask import Flask, request, jsonify, render_template

# Load the saved model
model = load("xgr_calories.joblib")
print(model)
# Initialize Flask app
app = Flask(__name__)
# Initialize Flask app

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.json
        gender = data['gender']  # 0 or 1
        heartrate = data['heartrate']
        body_temp = data['body_temp']
        height = data['height']
        age=data['age']
        
        # Prepare the input for the model
        inputs = [gender, age,height,heartrate, body_temp]
        inputs=[[float(i)for i in inputs]]
        print(inputs)
        # Make prediction
        calories = model.predict(inputs)[0]
        print(calories)
        
        # Return the prediction
        return jsonify({'calories': float(calories)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
