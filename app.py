from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, static_url_path='/static')

model = pickle.load(open("model.pkl","rb"))


@app.route('/', methods=["POST", "GET"])
def home():
    result = ""
    print(request.method)
    if request.method == 'POST':
        pm10 = request.form["pm10"] 
        pm25 = request.form["pm25"] 
        pm_sulfida = request.form["pm_sulfida"] 
        co2 = request.form["co2"]
        o = request.form["o"] 
        nitrogen = request.form["nitrogen"]

        data = [np.array([pm10, pm25, pm_sulfida, co2, o, nitrogen])]
        scaler = pickle.load(open("minmax.pkl","rb"))
        data = scaler.transform(data)
        result = model.predict(data)
        result = result[0]
    return render_template("index.html",result = result)


if __name__ == '__main__':
    app.run(debug=True)