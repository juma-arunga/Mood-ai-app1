import joblib

model = joblib.load("mood_model.pkl")

while True:
    text = input("Enter how you feel (or 'quit'): ")
    if text.lower() == "quit":
        break

    prediction = model.predict([text])
    print("Predicted mood:", prediction[0])
    