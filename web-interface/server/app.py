from flask import Flask, render_template, request

# Back-end 
# computes based on input from frontend, then sends back to front end
app = Flask(__name__)

# Home Page
@app.route("/")
def hello():
    return render_template('index.html')

# prediction page
@app.route("/predict", methods=['POST'])
def predict():
    # use requests to get data from the form
    prompt = request.form['prompt']

    # Make a prediction from some model, give back
    output = prompt

    # Test using text – change output so that the params are actually audio and preset files
    return render_template('index.html', prediction_text=f'Normally, this would be the bit where we give you audio, etcetc, but thats not done yet. Anyway, you said: `{output}\', nice!')

if __name__ == "__main__":
    app.run()