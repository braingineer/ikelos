from flask import Flask, render_template, request
import json
from sqlitedict import SqliteDict
app = Flask(__name__)


@app.route("/publish/parameters/", methods=['POST'])
def handle_params():
	data = json.loads(request.form['data'])
	name = data.pop("name")
	new_str = "<div class='parameters'><h1>{}</h1>".format(name)
	for k,v in data.items():
		new_str += "<div class='param_item'><b>{}</b> ::: {}</div>".format(k, v)
	new_str += "</div>"
	with SqliteDict("web.db") as db:
		db['experiment'] = new_str
		print("saving to db =)")
		db['logs'] = []
		db.commit()
	return '', 204

@app.route('/publish/epoch/end/', methods=['POST'])
def handle_data():
	data = json.loads(request.form['data'])

	epoch = data.pop("epoch")
	new_str = "<div class='epoch_data'><h2>epoch {}</h2>".format(epoch)
	new_str += "<br><p>"
	for k,v in data.items():
		new_str += "<div class='epoch_data_item'><b>{}</b> ::: {}</div>".format(k, v)
	new_str += "</div>"
	with SqliteDict('web.db') as db:
		logs = db['logs']
		logs.append(new_str)
		db['logs'] = logs
		db.commit()
	return '', 204

@app.route("/")
@app.route('/monitor')
def monitor():
	with SqliteDict('web.db') as db:
		exp = db['experiment'] if 'experiment' in db else "None currently"
		logs = '\n'.join(db['logs'][::-1]) if 'logs' in db else ""
		return """
		<h1>ada experiment page</h1>
		{}
		{}
		""".format(exp, logs)

if __name__ == "__main__":
	app.run(port=11031, debug=True)