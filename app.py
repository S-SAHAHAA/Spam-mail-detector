from flask import Flask, request, render_template
import pickle
import joblib
import numpy as np

app = Flask(__name__,template_folder='templates')

# Load your trained model and vectorizer
with open('modelv.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('modelj.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_features = vectorizer.transform([input_text])
        prediction = model.predict(input_features)

        if prediction[0] == 1:
            result = "Spam"
        else:
            result = "Not Spam"

        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


