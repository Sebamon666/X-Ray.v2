from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
modelo = joblib.load(r"D:\Python\BC IA\7. Técnicas avanzadas y empleabilidad\Clase 27\Titanic.pkl")

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Predicción Titanic</title>
</head>
<body>
    <h2>Predicción Titanic</h2>
    <form action="/predict_web" method="post">
        Edad: <input type="number" name="Age" value="34"><br><br>
        Pclass: <input type="number" name="Pclass" value="3"><br><br>
        Fare: <input type="text" name="Fare" value="7.8292"><br><br>
        Parch: <input type="number" name="Parch" value="0"><br><br>
        SibSp: <input type="number" name="SibSp" value="0"><br><br>
        C: <input type="checkbox" name="C"><br><br>
        Q: <input type="checkbox" name="Q" checked><br><br>
        S: <input type="checkbox" name="S"><br><br>
        Sexo:<br>
        <input type="radio" name="female" value="1"> Mujer
        <input type="radio" name="male" value="1" checked> Hombre
        <br><br>
        <input type="submit" value="Predecir">
    </form>
    {% if prediction is not none %}
        <h3>Resultado: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html, prediction=None)

@app.route("/predict_web", methods=["POST"])
def predict_web():
    # recoger valores del formulario
    data = {
        'Age': float(request.form.get('Age', 0)),
        'C': 'C' in request.form,
        'Fare': float(request.form.get('Fare', 0)),
        'Parch': int(request.form.get('Parch', 0)),
        'Pclass': int(request.form.get('Pclass', 0)),
        'Q': 'Q' in request.form,
        'S': 'S' in request.form,
        'SibSp': int(request.form.get('SibSp', 0)),
        'female': 'female' in request.form,
        'male': 'male' in request.form,
    }
    columnas = ['Age','C','Fare','Parch','Pclass','Q','S','SibSp','female','male']
    df_prueba = pd.DataFrame([[data[col] for col in columnas]], columns=columnas)

    pred = modelo.predict(df_prueba)
    return render_template_string(html, prediction=int(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)
