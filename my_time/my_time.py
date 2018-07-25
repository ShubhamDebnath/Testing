import csv
import time
import datetime
from win32gui import GetWindowText, GetForegroundWindow

def return_time():
	now = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
	return now

csv_file = open('output/recorded_times.csv', 'a', newline = '')
writer = csv.writer(csv_file)

headers = ['Window', 'Start_time', 'End_time', 'Duration']
writer.writerow(headers)

window = GetWindowText(GetForegroundWindow())
begin = time.time()
start_time = return_time()
end_time = return_time()
duration = 0

while True:
	try:
	    new_window = GetWindowText(GetForegroundWindow())

	    if new_window != window and len(new_window) > 0:
		    curr = time.time()
		    duration = curr - begin
		    end_time = return_time()
		    row = [window, start_time, end_time, duration]
		    writer.writerow(row)
		    start_time = end_time
		    window = new_window
		    begin = time.time()

	except:
		pass