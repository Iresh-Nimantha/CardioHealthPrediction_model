from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

# Load model
model = joblib.load("HartDiseaseRisk.sav")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_result():
    # Collect form data
    name = request.form['name']
    gender = float(request.form['gender'])
    age = float(request.form['age'])
    tc = float(request.form['tc'])
    hdl = float(request.form['hdl'])
    smoke = float(request.form['smoke'])
    bpm = float(request.form['bpm'])
    diab = float(request.form['diab'])

    # Prepare data for prediction
    test_data = np.array([gender, age, tc, hdl, smoke, bpm, diab]).reshape(1, -1)
    prediction = model.predict(test_data)

    # Round result
    risk_score = round(prediction[0], 2)

    # Redirect to report page with data
    return render_template(
        'report.html',
        name=name,
        gender=int(gender),
        age=age,
        tc=tc,
        hdl=hdl,
        smoke=int(smoke),
        bpm=int(bpm),
        diab=int(diab),
        resulte=risk_score
    )

if __name__ == '__main__':
    app.run(debug=True)
