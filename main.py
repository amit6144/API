from flask import Flask, jsonify, request
import pickle
import numpy as np
import pandas as pd

# creating a Flask app
app = Flask(__name__)
model=pickle.load(open('CreditScore.pkl','rb'))

# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods=['GET'])
def home():
    data = "hello world"
    return jsonify({'data': data})


# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@app.route('/predict', methods=['POST'])
def disp():



    loanstatus=request.form.get("Loan Status")
    loanAmount=request.form.get("Current Loan Amount")
    annualIncome=request.form.get("Annual Income")
    creditbalance=request.form.get("Current Credit Balance")
    result={'1':loanstatus,'2':annualIncome,'3':loanAmount,'4':creditbalance}

    test=np.array([int(loanstatus),float(loanAmount),float(annualIncome),float(creditbalance)]).reshape(1,-1)

    res=model.predict(test)[0][0]

    return ({'result':str(res)})

# driver function
if __name__ == '__main__':
    app.run(debug=True)
