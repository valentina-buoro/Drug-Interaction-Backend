import os
from flask import Flask
from flask_cors import CORS


def create_app(test_config=None,*args, **kwargs):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def hello():
        return '<h1>Welcome to the Drug Interaction Predictor</h1><p>Enter your SMILES strings below.</p>'

    from . import drug_interaction
    app.register_blueprint(drug_interaction.bp)

    return app