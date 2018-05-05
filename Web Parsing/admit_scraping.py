import urllib.request
import os

# just to practive urlretrieve
url = 'http://HIDDEN IP ADDRESS OF SERVER/jasperserver/flow.html?rollno={0}&_flowId=viewReportFlow&reportUnit=%2Freports%2FMAKAUT%2FPRE_REPORT%2FAdmit_Card&standAlone=true&ParentFolderUri=/reports/MAKAUT/PRE_REPORT&j_username=LSPL&j_password=lspl12345&decorate=no&output=pdf&semcode=SM{1:2f}'
file_name = '{0}/Admit Card {1}.pdf'
baseRollNo = {'Computer Science 6th Sem' : XXXXXXXXXXX,      # add more roll numbers to this list, keep last 2 digits 00
                'Electrical 6th Sem' : XXXXXXXXXXX ,
                'Civil 6th Sem' : XXXXXXXXXXX ,
                'ECE 6th Sem' : XXXXXXXXXXX
                }                                           

for dept, roll in baseRollNo.items():
    for i in range(1,100):
        try:
            if not os.path.exists(dept):
                os.makedirs(dept)
            curr_roll = roll + i
            urllib.request.urlretrieve(url.format(curr_roll, 6), file_name.format(dept, curr_roll))    # just notice this line
        except Exception as e:
            pass
    print('done for ' + dept)
print('all done')
