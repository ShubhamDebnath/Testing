from subprocess import Popen, PIPE
from default_code import return_default_code, return_extension, return_compiler, return_os
from flask import Flask, render_template, request


app = Flask(__name__)

fileName = 'output/Main.{}'

choice = "c"
code = return_default_code(choice)
command_separator = return_os()
default_output = "Your output appears here"
default_stdin = "Put standard input here"

@app.route("/")
def home():
	return render_template('home.html', choice = choice, default_code = code, default_output = default_output, default_stdin = default_stdin)

# @app.route("/", methods = ["POST"])
# def select_language():
# 	global choice
# 	if 'lang' in request.form.values():
# 	    choice = request.form["lang"]
# 	    # return choice
# 	    code = return_default_code(choice)
# 	return render_template('home.html', choice = choice, default_code = code, default_output = default_output, default_stdin = default_stdin)


@app.route("/", methods = ["POST"])
def home_post():
	global choice, code

	# if request.form["lang"] in ['c', 'c++', 'c#', 'java', 'javascript', 'python3']:
	if 'submit' not in request.form.values():
		# print('*************************LANG')
		choice = request.form["lang"]
		code = return_default_code(choice)
		return render_template('home.html', choice = choice, default_code = code, default_output = default_output, default_stdin = default_stdin)

	elif 'submit' in request.form.values():
		# print("*********************submit\n\n")
		code = request.form["code"]
		stdin = request.form["stdin_string"]
		extension = return_extension(choice)
		compiler = return_compiler(choice)
		save_file(code, extension)
		out, err = execute_file(extension, compiler, stdin)
		if err.decode("utf-8"):
			return render_template('home.html', choice = choice, default_code = code, default_output = err.decode("utf-8"), default_stdin = stdin)
		else:
			return render_template('home.html', choice = choice, default_code = code, default_output = out.decode("utf-8"), default_stdin = stdin)

	else:
		return render_template('home.html', choice = choice, default_code = code, default_output = default_output, default_stdin = default_stdin)

def save_file(code, extension):	
	file = open(fileName.format(extension), 'w', newline = '')
	file.write(code)
	file.close()

def execute_file(extension, compiler, stdin = None):
	if extension not in ['c', 'cpp', 'java']:
		p = Popen([compiler, fileName.format(extension)], stdin=PIPE, stdout=PIPE, stderr=PIPE)
	elif 'g' in compiler:
		p = Popen([compiler, fileName.format(extension), '-o', 'output/Main', command_separator, "output\\Main.exe"], stdin = PIPE, stdout = PIPE, stderr = PIPE, shell = True)
	else:
		p = Popen([compiler, fileName.format(extension), command_separator, "java", "-cp", "output", "Main"], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell = True)
	output, err = p.communicate(input = stdin.encode('utf-8'))
	# print("output : ", output ,"\nerror :",  err)
	return output, err


if __name__ == '__main__':
	app.run(debug = True)