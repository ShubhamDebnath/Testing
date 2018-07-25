import csv
from flask import Flask, render_template, request

app = FLask(__name__)
fileName = 'output/recorded_times.csv'

def sum_duration(file):
    csv_file = open(file, 'r')
    reader = csv.reader(csv_file, delimiter = ',')
    data = {}

    # skipping the headers
    row = next(reader)

    for row in reader:
        window = row[0]
        duration = float(row[3])

        # print(list(data))
        # print(data)

        if window in list(data):
            data[window] += duration
        else:
            data[window] = duration

    csv_file.close()

    return data

@app.route('/')
def home():
	return render_template('home.html', )


# data = sum_duration(fileName)
# for window, duration in data.items():
    # print(window + '\t' + str(duration))

if __name__ == '__main__':
	app.run(debug = True)