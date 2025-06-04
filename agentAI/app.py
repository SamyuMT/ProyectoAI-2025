# Suggested code may be subject to a license. Learn more: ~LicenseLog:2777148287.
# Suggested code may be subject to a license. Learn more: ~LicenseLog:3854100564.
import os
import joblib
from flask import Flask
from agent import *
from flask import request
from dotenv import load_dotenv
from flask_swagger_ui import get_swaggerui_blueprint
from flask import jsonify
import pandas as pd
import time

# Suggested code may be subject to a license. Learn more: ~LicenseLog:3854100564.
app = Flask(__name__)

api_key = os.getenv("DEEPSEEK_API_KEY")

# Swagger configuration
SWAGGER_URL = '/docs'
API_URL = '/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Agent-Ceed API"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/swagger.json')
def swagger_spec():
    """
    Generates the Swagger JSON file automatically from Flask routes.
    """
    from flask_swagger import swagger
    swag = swagger(app)
    swag['info'] = {
        "title": "Agent-Ceed API Documentation",
        "version": "1.0",
        "description": "Documentation for the Agent-Ceed API with all registered routes"
    }
    return jsonify(swag)

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"

@app.route('/get_pred', methods=['POST'])
def get_prediction():
    """
    Esta ruta permite obtener una predicción enviando un archivo PDF a través de una solicitud POST.

    ---
    parameters:
      - name: pdf
        in: formData
        type: file
        required: true
        description: El archivo PDF para la predicción.
    responses:
      200:
        description: Predicción obtenida exitosamente. Devuelve un payload con el resultado.
        schema:
          type: object
          properties:
            data:
              type: number
              description: El valor de la predicción reescalada.
            status:
              type: string
              description: El estado de la operación (siempre "ok" si es exitoso).
      400:
        description: No se proporcionó un archivo PDF o el archivo seleccionado está vacío.
    """
    start_time = time.time()

    if 'pdf' not in request.files:
        return "No PDF file provided", 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return "No selected file", 400

    if pdf_file:
        load_start = time.time()
        preproc   = joblib.load("logs/feature_pipeline.joblib")
        y_scaler  = joblib.load("logs/y_scaler.joblib")
        model     = joblib.load("logs/LGBM_Full_model.joblib")
        load_time = time.time() - load_start
        print(f"Model loading time: {load_time:.2f} seconds")

        # Use the file object directly instead of saving it
        process_start = time.time()
        agent = AgentEstimacionCEED(api_key)
        result = agent.process_ceed_form(pdf_file)
        process_time = time.time() - process_start
        print(f"PDF processing time: {process_time:.2f} seconds")

        pred_start = time.time()
        result = json.loads(result)
        X_new = pd.DataFrame([result])
        X_new_trans   = preproc.transform(X_new)
        y_pred = model.predict(X_new_trans)
        y_pred_rescaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_pred_rescaled = y_pred_rescaled.flatten()
        y_pred_rescaled = y_pred_rescaled * 1000
        y_pred_rescaled = y_pred_rescaled.round(2)
        pred_time = time.time() - pred_start
        print(f"Prediction time: {pred_time:.2f} seconds")


        text = agent.mensaje_lindo(y_pred_rescaled)
        total_time = time.time() - start_time
        print(f"Total request time: {total_time:.2f} seconds")

        payload = {
            "data": text,
        }
        return payload, 200


if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8100)))