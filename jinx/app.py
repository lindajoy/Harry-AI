from flask import Flask, jsonify, request
from flask_cors import CORS
from main import main

app = Flask(__name__)
CORS(app)  

"""
    Input here should be: The request prompt, and pass it to the model
"""
@app.route('/api/ask-question', methods=['POST'])
def generateResponse():
    data = request.get_json()
    form_value = data.get('formValue', {})  
    result = main(form_value) 
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
