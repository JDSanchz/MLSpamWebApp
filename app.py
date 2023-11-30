from flask import Flask, request, render_template
import pickle

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the message from the submitted form
        message = request.form['message']
        # Transform the message for the model
        message_transformed = vectorizer.transform([message]).toarray()
        # Predict using the model
        prediction = model.predict(message_transformed)
        # Determine whether it's spam or ham
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        # Render the result template with the prediction
        return render_template('result.html', message=message, result=result)
    # Render the index template for GET request
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
