from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences
from urllib.parse import urlparse
from web3 import Web3
import os
import json

app = Flask(__name__)
CORS(app)

# Load the trained Keras model
model = load_model('./model_gru.h5')

# Load the tokenizer
with open('.tokenizer.json', 'r') as file:
    tokenizer_json = file.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Define maxlen (should match the value used during training)
maxlen = 200

# Infura and Web3 setup
infura_api_key = os.getenv('INFURA_API_KEY')
provider_url = f"https://polygon-mainnet.infura.io/v3/{infura_api_key}"
web3 = Web3(Web3.HTTPProvider(provider_url))

contract_address = Web3.to_checksum_address("0x45b47d4a68babd0286ab3ee75bbcd23986516760")
contract_abi = [
    {
        "inputs": [{"internalType": "string", "name": "url", "type": "string"}],
        "name": "addUrl",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "urlEntries",
        "outputs": [
            {"internalType": "string", "name": "url", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getUrlCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

private_key = os.getenv('PRIVATE_KEY')
account_address = web3.eth.account.from_key(private_key).address
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

def preprocess(url):
    """
    Preprocess the URL into features compatible with the model.
    Tokenize and pad the URL for Keras model input.
    """
    url_sequence = tokenizer.texts_to_sequences([url])  # Tokenize
    url_padded = pad_sequences(url_sequence, maxlen=maxlen)  # Pad
    return url_padded

def is_url_blocked(url):
    try:
        url_count = contract.functions.getUrlCount().call()
        for i in range(url_count):
            entry = contract.functions.urlEntries(i).call()
            if entry[0] == url:
                return True
        return False
    except Exception as e:
        print(f"Error checking URL in contract: {e}")
        return False

def add_url_to_block(url):
    try:
        nonce = web3.eth.get_transaction_count(account_address)
        transaction = contract.functions.addUrl(url).build_transaction({
            'gas': 2000000,
            'gasPrice': web3.to_wei('50', 'gwei'),
            'nonce': nonce,
            'from': account_address
        })
        signed_txn = web3.eth.account.sign_transaction(transaction, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
        web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"URL {url} added to block successfully")
    except Exception as e:
        print(f"Error adding URL to block: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    
    if url is None:
        return jsonify({'message': 'URL is required'}), 400

    if is_url_blocked(url):
        return jsonify({'message': 'URL is already blocked', 'isFake': True})

    # Preprocess the URL
    features = preprocess(url)
    
    # Make prediction using the Keras model
    prediction = model.predict(features)
    is_fake = prediction[0][0] > 0.5  # Binary classification threshold

    if is_fake:
        add_url_to_block(url)

    result = {
        'isFake': bool(is_fake)
    }
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
