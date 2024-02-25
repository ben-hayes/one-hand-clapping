from flask import Flask, render_template, request

# TODO: #4 Load vst: Dropdown or upload
# TODO: #5 Returns: Audio, preset file
# TODO: #10 Controls for VST: Add popup/alert to control and load presets from json
# TODO: #6 Controls for algorithm: Population size, max_generations
# TODO: #7 Interface: bulma

# Back-end 
# Computes parameters and sound based on input from frontend, then returns it
app = Flask(__name__)


# Placeholder functions
Placeholder_ID = {"name": "Default VST Name"}

def get_vst():
    return Placeholder_ID

def set_vst(new_ID):
    Placeholder_ID = new_ID
    return

def get_params():
    return {
        "attack": 0,
        "Decay": 1,
        "Sustain": 1,
        "Release": 1
        }

def set_params(new_params):
    return

# Placeholders done


# # Create all defaults etc for page, before rendering prediction/sending page info
# def instantiatePage():
#     # Get vst details from the backend
#     return get_vst(), list_params()


# Get and set VST at this address
# 'GET', 
@app.route("/vst", methods=['GET'])
def vst():
    # return request.get_data
    return get_vst()

# Load and config VST
@app.route("/config", methods=['GET', 'POST'])
def load():
    # set_vst(request.form['vsts']) 
    return get_params()

# Prediction Page
@app.route("/predict", methods=['POST'])
def predict():
    # Get data from the form
    prompt = request.form['prompt']

    # Make a prediction from some model, give back
    output = prompt

    # Test using text – change output so that the params are actually audio and preset files
    return f'Normally, this would be the bit where we give you audio, etcetc, but thats not done yet. Anyway, you said: `{output}\', nice!'

if __name__ == "__main__":
    app.run()