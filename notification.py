import smtplib
import os
from email.message import EmailMessage
import config
import traceback
import sys
from argparse import ArgumentParser


def send_gmail_notification(email_content: str, mail_subject: str, to_addr: str) -> None:
    # collect mail params
    mail_dict = {}
    for mail_key in ['MAIL_USER', 'MAIL_APP_PASS']:
        if mail_key in os.environ.keys():
            mail_dict[mail_key] = os.environ[mail_key]
        else:
            raise ValueError(f"Missing environmental parameter {mail_key} required for authentication. Exiting")
    try:
        with open(email_content) as fp:
            msg = EmailMessage()
            msg.set_content(fp.read())
        msg['Subject'] = mail_subject
        msg['From'] = f"{mail_dict['MAIL_USER']}@gmail.com"
        msg['To'] = to_addr if to_addr else f"{mail_dict['MAIL_USER']}@gmail.com"
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(mail_dict['MAIL_USER'], mail_dict['MAIL_APP_PASS'])
        server.sendmail(msg['From'], msg['To'], msg.as_string())
        server.close()
        print('notification email sent')
    except Exception as e:  # a lot could go wrong here. for now, shamefully using a broad except/logging traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"error sending notification email: "
              f"{repr(traceback.format_exception(exc_type, exc_value, exc_traceback))}")
        raise e


def main():
    parser = ArgumentParser(description="""\
Sends a gmail email using a from email and password specified via env variables MAIL_USER and MAIL_APP_PASS respectively
""")
    parser.add_argument('-f', '--file', required=True, dest='email_content', help="""file to send in the email""")
    parser.add_argument('-s', '--subject', dest='mail_subject', default=config.mail_subject, help='The email subject')
    parser.add_argument('-t', '--to', required=False, dest='to_addr', default=None, help='recipient of email')
    args = parser.parse_args()
    send_gmail_notification(args.email_content, args.mail_subject, args.to_addr)


if __name__ == '__main__':
    main()