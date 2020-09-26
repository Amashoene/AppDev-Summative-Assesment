import csv
import flask
import dashboard


server = flask.Flask(__name__)
app = dashboard.get_dash(server)

@server.route("/")
def index():
    # go back to home page
    return flask.render_template("index.html")


@server.route("/call_back_1")
def call_back_1():
    print("call back used.")

    # go back to home page
    return flask.redirect("/")

@server.route("/read_csv")
def read_csv():
    with open("maintenancefile.csv") as wfile:
        reader = csv.reader(wfile)
        lines = []
        for line in reader:
            print(line)
            lines.append(line)
        return {
            "lines" : lines
        }
        
        
@server.route("/store_file", methods = ["POST"])
def store_file():
    #storing the file here 
    file_obj = flask.request.files["filename"]
    print(file_obj)
    file_obj.save("maintenancefile.csv")
    return "Upload finished"


if __name__ == '__main__':
    app.run_server(port=5001, debug=True)
   

if __name__ == '__main__':
    app.run_server(port=5001, debug=True)