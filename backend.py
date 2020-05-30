from flask import Flask, request
from classifier import interpret, classify
import json

app = Flask (__name__)

@app.route("/")
@app.route("/classify", methods=["GET"])
def predict():
    url = request.args["img_url"]

    return json.dumps(interpret(classify(url)))

    # if request.method == "POST":
    #     return classify()

# then send a post request to localhost:5000/?img_url=<IMAGE_URL>
# curl https://applorange.herokuapp.com/?img_url=<IMAGE_URL>

if __name__ == "__main__":
    app.run(debug=True)