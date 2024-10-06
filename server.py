from flask import Flask, request, jsonify
from preds import get_preds
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route("/")
def ping():
    return "Server pinged successfully"

@app.route("/predict", methods=["GET"])
def pred():
    try:
        # Await the async function get_preds
        preds = get_preds()

        preds_json = preds.to_json(orient="records")

        return preds_json
    except Exception as e:
        return jsonify({"error": str(e)}), 500
