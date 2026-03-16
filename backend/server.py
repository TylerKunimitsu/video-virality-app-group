# This file is for connecting the frontend and backend (API server)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Import your existing prediction pipeline
from predict import predict_virality

app = Flask(__name__)
CORS(app) # This allows your React frontend (port 3000 or 5173) to talk to Flask (port 5000)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Grab the text data from the frontend request
        title = request.form.get('title', '')
        description = request.form.get('description', '')
        tags = request.form.get('tags', '')
        
        # 2. Grab the uploaded image
        if 'thumbnail' not in request.files:
            return jsonify({"error": "No thumbnail uploaded"}), 400
            
        file = request.files['thumbnail']
        
        # 3. Save the image temporarily so predict.py can read it
        temp_image_path = "temp_upload.jpg"
        file.save(temp_image_path)
        
        # 4. Run your prediction!
        results = predict_virality(title, description, tags, temp_image_path)
        
        # 5. Clean up the temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        # 6. Send the results back to React
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting prediction server on http://localhost:5000")
    app.run(debug=True, port=5000)