from flask import Flask, request
from classifier import interpret, classify
import json

app = Flask (__name__)

@app.route("/", methods=["GET"])
def predict():
    if request.args:
        url = request.args["img_url"]
        return json.dumps(interpret(classify(url)))
    else:
        return "Use the ?img_url= argument to choose an image by url"

# then send a GET request to localhost:5000/?img_url=<IMAGE_URL>
# curl https://applorange.herokuapp.com/?img_url=<IMAGE_URL>

if __name__ == "__main__":
    app.run(debug=True)