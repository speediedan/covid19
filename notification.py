import smtplib
from typing import List
from pathlib import Path
import os
from email.message import EmailMessage
import sys


with open(textfile) as fp:
    # Create a text/plain message

    msg.set_content(fp.read())

# me == the sender's email address
# you == the recipient's email address
msg['Subject'] = f'The contents of {textfile}'
msg['From'] = me
msg['To'] = you


sent_from = gmail_user
to = ['me@gmail.com', 'bill@gmail.com']
subject = 'OMG Super Important Message'
body = 'Hey, what's up?\n\n- You'

email_text = """\
From: %s
To: %s
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)
def send_gmail_notification(subject, from_addr, to_addr, email_content):
    # collect mail params
    mail_dict = {}
    for mail_key in ['MAIL_USER', 'MAIL_APP_PASS']:
        mail_dict[mail_key] = os.environ[mail_key] if mail_key in os.environ.keys() else None
    try:
        msg = EmailMessage()
        msg['Subject']
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()

        print 'Email sent!'
    except:
        print 'Something went wrong...'