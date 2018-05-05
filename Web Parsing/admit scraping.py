import urllib.request
import os

url = 'http://13.228.162.242/jasperserver/flow.html?rollno={0}&_flowId=viewReportFlow&reportUnit=%2Freports%2FMAKAUT%2FPRE_REPORT%2FAdmit_Card&standAlone=true&ParentFolderUri=/reports/MAKAUT/PRE_REPORT&j_username=LSPL&j_password=lspl12345&decorate=no&output=pdf&semcode=SM{1:2f}'
file_name = '{0}/Admit Card {1}.pdf'
baseRollNo = {'Computer Science 6th Sem' : 11900115000,      # add more roll numbers to this list, keep last 2 digits 00
                'Electrical 6th Sem' : 11901615000 
                'Civil 6th Sem' : 11901315000 
                'ECE 6th Sem' 11900315000:
                }                                           # baki IT 6th sem ka roll nahi mere pass 

for dept, roll in baseRollNo.items():
    for i in range(1,100):                                  # mujhe pata nahi har sem mein kitna kitna bacha hai, so last mein blank files banega, 
                                                            # though just ek extra line lagega , but too lazy
        try:
            if not os.path.exists(dept):
                os.makedirs(dept)
            curr_roll = roll + i
            urllib.request.urlretrieve(url.format(curr_roll, 6), file_name.format(dept, curr_roll))    # just yeh line lagta file download karne ko
        except Exception as e:
            pass
    print('done for ' + dept)
print('all done')