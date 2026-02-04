from flask import Flask, request, jsonify, render_template
#render_template for rendering the html file
#flask for creating a server
#request for getting data from user
#jsonify for sennding data to the user
import joblib
import re

model = joblib.load("mood_model.pkl")
#loading the trained model
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
    #cleaning the text
app = Flask(__name__)
#creating a server application
@app.route("/")
def home():
    return render_template("index.html")
  #home page,where the user will see the html file
@app.route("/predict", methods=["POST"])
#predict page,where the user will send the data and get the prediction 
def predict():
  data = request.get_json()
    #reading the data from the user
  if not data or "text" not in data:
        return jsonify({"error": "Please provide text"}), 400
    #validates the user inputs if its correct or not
  user_text = clean_text(data["text"])
  prediction = model.predict([user_text])[0]
    #clean the user text and predict the mood
  return jsonify({"mood": prediction})
  #send the prediction to the User
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
   #run the server
   