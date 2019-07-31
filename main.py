import argparse
import json

from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api
import numpy as np

from emotion_classifier import EmotionClassifier

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

classifier = None


class ClassifierResource(Resource):
    def put(self):
        req_body = request.get_json()
        print(req_body)
        hr = None
        if 'hr' in req_body:
            hr = np.asarray(req_body['hr'])
        emotion, additional = classifier.classify(np.asarray(req_body['rr']), hr)
        response = app.response_class(
            response=json.dumps({
                'emotion': emotion,
                'additional': json.dumps(additional)
            }),
            status=200,
            mimetype='application/json'
        )
        return response


api.add_resource(ClassifierResource, '/classify')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heart Rate, RR-interval emotion classifier')
    parser.add_argument('--classifier-path', required=True, help='Path to the classifier joblib')
    parser.add_argument('--preprocessing-path', default=None,
                        help='Path to the preprocessing joblib. If not given, no preprocessing will be done')
    args = parser.parse_args()
    classifier = EmotionClassifier(args.classifier_path,
                                   args.preprocessing_path)
    app.run(debug=True)
