from flask import Flask


app = Flask(__name__)


from controllers.LlamaController import llama_controller


app.register_blueprint(llama_controller, url_prefix='/')

app.run(debug=True)
