import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from .base_notifier import BaseNotifier

class EmailNotifier(BaseNotifier):
  """
  The :class:`.EmailNotifier` class can be used to send email notifications.
  """
  def __init__ (self, email_subject, sender_account, sender_password, smtp_server, smtp_port, recipients):
    """
    This is the constructor method and can be used to create a new object instance of :class:`.EmailNotifier` class.

    :param email_subject: The subject of the email.
    :type email_subject: str

    :param sender_account: The email account of the sender. For example, 'someone@gmail.com'.
    :type sender_account: str

    :param sender_password: The password of the sender email account.
    :type sender_password: str 

    :param smtp_server: The secured SMTP server of the sender email account. For example, for Gmail is 'smtp.gmail.com'.
    :type smtp_server: str

    :param smtp_port: The port of the secured SMTP server. For example, for Gmail is 465.
    :type smtp_port: int

    :param recipients: The recipients (email addresses) as CSV.
    :type recipients: str
    """
    self.recipients = [ o.strip() for o in recipients.split(',') ]

    self.email_subject = email_subject

    self.sender_account = sender_account

    self.sender_password = sender_password

    self.smtp_server = smtp_server

    self.smtp_port = smtp_port

  def notify(self, message):
    """
    Send the email notification.

    :param message: The notification's message.
    :type message: str
    """

    server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)

    server.ehlo()

    server.login(self.sender_account, self.sender_password)

    for recipient in self.recipients:
      email = MIMEMultipart('alternative')

      email['From'] = self.sender_account
      email['To'] = recipient
      email['Subject'] = self.email_subject

      email.attach(MIMEText(message, 'html'))

      text = email.as_string()

      server.sendmail(self.sender_account, recipient, text)

    server.quit()